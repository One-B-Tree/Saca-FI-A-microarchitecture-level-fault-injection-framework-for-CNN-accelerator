import math 
from tqdm import tqdm


def sram_traffic(
        dimension_rows=4,#pe参数行，下面是pe参数列
        dimension_cols=4,
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="RS/sram_read.csv",
        sram_write_trace_file="RS/sram_write.csv"
    ):

    # Dimensions of output feature map channel输出特征矩阵尺寸3,3
    E_h = math.floor((ifmap_h - filt_h + strides) / strides)
    E_w = math.floor((ifmap_w - filt_w + strides) / strides)
    
    # Number of pixels in one convolution window 一个卷积核像素数54
    px_per_conv_window = filt_h * filt_w * num_channels
    r2c = px_per_conv_window

    # Total number of ofmap px across all channels 输出矩阵的像素总数
    num_ofmap_px = E_h * E_w * num_filt
    e2  = E_h * E_w#一片输出矩阵像素数
    e2m = num_ofmap_px
    
    # Variables to calculate folds in runtime
    num_h_fold = 1
    num_v_fold = 1 
    max_parallel_window = 1

    # Variables for utilization calculation
    util = 0
    compute_cycles = 0
    #这里计算卷积核是否需要切分。。。就是行数是否满足要求
    if dimension_rows < px_per_conv_window:
        num_h_fold = math.ceil(px_per_conv_window/dimension_rows)
        #print('在，num_h_fold值为',num_h_fold)
    else:#行刚好满足，不需要切分
        max_parallel_window = math.floor(dimension_rows/ px_per_conv_window)
        #print('在，max_parallel_window值为', max_parallel_window)
    reqd_cols = num_filt                    # Total number of cols to be mapped卷积核总数
    max_cols_per_v_fold = max_parallel_window * dimension_cols
    num_v_folds = math.ceil(reqd_cols / max_cols_per_v_fold)
    #print("在，num_v_folds:",num_v_folds)
    remaining_cols = reqd_cols
    cycles = 0
    prev_cycl = 0

    #print("Vertical folds = " +str(num_v_folds))
   
    # These are the starting addresses of filter weights in the memory 过滤器权重起始地址
    all_col_addr_list = []
    for c in range(num_filt):
        addr = (c) * r2c + filt_base
        all_col_addr_list.append(addr)
        #print("addr:", addr)#100000,100018,1000036为什么只加了这几个？

    # These are the starting addresses of ifmap windows in the memory ifmap窗口地址
    hc = ifmap_w * num_channels
    all_ifmap_base_addr = []
    for px in range(int(e2)):         #number of ofmap px in a ofmap channel
        addr = (px / E_w) * strides * hc + (px%E_w) * strides
        all_ifmap_base_addr.append(addr)
        #print("addr:", addr)#这里面我完全不明白加到数组里的都是什么东西
    for v in tqdm(range(int(num_v_folds))):
        #print("V fold id: " + str(v))
            
        # Take a slice of the starting addresses that are relevant for this v_fold 
        cols_this_fold = min(remaining_cols, max_parallel_window * dimension_cols)
        idx_start = v * dimension_cols
        idx_end = idx_start + cols_this_fold
        col_addr_list = all_col_addr_list[idx_start:idx_end]
        #print("col_addr_list在这，顺便看看循环多少次",col_addr_list)#根本只是执行了1次，而且数据也没有改变，计算折叠的是真的麻烦，先算没有折叠的简化一下计算
        if num_h_fold > 1 :
           
            rem_h = r2c                     # Tracks the elements processed within a conv filter 
            next_ifmap_addr = ifmap_base    # Starts from the top left corner of the IFMAP matrix

            for h in range(num_h_fold):
                rows_this_fold = min(rem_h, dimension_rows) 
                #print("h fold id: " + str(h))

                # Values returned
                # cycles        -> Cycle count for the next operation ie. cycles elapsed + 1
                # col_addr_list -> The starting filter address for the next iteration
                cycles, col_addr_list   = gen_trace_filter_partial(
                                            col_addrs   = col_addr_list,
                                            cycle       = cycles,
                                            num_rows    = dimension_rows,
                                            remaining   = rows_this_fold,
                                            sram_read_trace_file = sram_read_trace_file
                                            )
                #print("Weights loaded by " + str(cycles) + " cycles")
                data_out_cycles     = cycles    #Store this cycle for parallel readout
                cycles_ifmap            = gen_trace_ifmap_partial(
                                            cycle = cycles,
                                            num_rows = dimension_rows, num_cols = dimension_cols,
                                            num_filters = num_filt,
                                            remaining = rem_h,
                                            remaining_filters = remaining_cols, 
                                            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                                            filt_h = filt_h, filt_w = filt_w,
                                            num_channels = num_channels,
                                            stride = strides, ifmap_base = ifmap_base,
                                            sram_read_trace_file = sram_read_trace_file
                                            )
                cycles_ofmap        = gen_trace_ofmap(
                                            cycle = data_out_cycles,
                                            num_rows = dimension_rows,
                                            num_cols = dimension_cols,
                                            ofmap_base = ofmap_base,
                                            window_size= rows_this_fold,
                                            parallel_window =1,
                                            num_ofmap_px = int(e2),
                                            filters_done = (v * dimension_cols),
                                            num_filter = num_filt,
                                            sram_write_trace_file = sram_write_trace_file
                                            ) 

                #print("IFMAPS processed by " + str(cycles) + " cycles")
                util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)

                rem_h -= rows_this_fold
                cycles = max(cycles_ifmap, cycles_ofmap)

                del_cycl = cycles - prev_cycl
                util += util_this_fold *  del_cycl
                compute_cycles += del_cycl
                prev_cycl = cycles

        else:
            #filters_this_fold = min(remaining_cols, max_cols_per_v_fold)
            filt_done = v * max_parallel_window * dimension_cols
            rem = num_filt - filt_done

            parallel_window = math.ceil(rem / dimension_cols)
            parallel_window = int(min(max_parallel_window, parallel_window))
            #print("col_addr,parallel_window,filters_this_fold分别是：",col_addr_list,parallel_window,cols_this_fold)
            #[1000000, 1000018, 1000036] 1 3为值
            cycles_filter = gen_filter_trace(
                                cycle = cycles,
                                num_rows = dimension_rows, num_cols = dimension_cols,
                                filt_h = filt_h, filt_w = filt_w, num_channels = num_channels,
                                col_addr = col_addr_list, 
                                parallel_window=parallel_window,
                                filters_this_fold=cols_this_fold,
                                sram_read_trace_file=sram_read_trace_file
                                )

            cycles_ifmap, rows_this_fold\
                            = gen_ifmap_trace(
                            cycle = cycles_filter,
                            num_rows = dimension_rows, num_cols = dimension_cols,
                            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                            filt_h = filt_h, filt_w = filt_w,
                            num_channels = num_channels, stride = strides,
                            parallel_window = parallel_window,
                            sram_read_trace_file = sram_read_trace_file
                            )

            cycles_ofmap = gen_trace_ofmap(
                            cycle = cycles_filter,
                            num_rows = dimension_rows, num_cols = dimension_cols,
                            ofmap_base = ofmap_base, 
                            parallel_window = parallel_window,
                            window_size = r2c,
                            num_ofmap_px = int(e2),
                            filters_done = int(v * max_parallel_window * dimension_cols),
                            num_filter = num_filt,
                            sram_write_trace_file = sram_write_trace_file
                            )
            cycles = max(cycles_ifmap, cycles_ofmap)
            del_cycl = cycles - prev_cycl

            # Since multiple filters are being mapped on a single col due to large number of rows
            # util calculation is a little involved,
            # cols_this_fold --> number of filters mapped this fold
            rem = cols_this_fold
            tmp_util = 0
            for _ in range(parallel_window):
                col_used = min(rem, dimension_cols)
                row_used = r2c                      # Number of row used will always be in multiple of r2c,
                                                    # parallel window calc took care of this
                tmp_util += row_used * col_used
                rem -= col_used

            #util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)
            util_this_fold = tmp_util /(dimension_rows * dimension_cols)
            util += util_this_fold * del_cycl
            compute_cycles += del_cycl
            prev_cycl = cycles

        remaining_cols -= cols_this_fold

    final = str(cycles)
    final_util = (util / compute_cycles) * 100
    #print("Compute finished at: " + str(final) + " cycles")
    return (final, final_util)


def gen_filter_trace(
        cycle = 0,
        num_rows = 4, num_cols = 4,
        filt_h = 3, filt_w = 3, num_channels = 3,
        col_addr = [],
        parallel_window = 1,
        filters_this_fold = 4,
        sram_read_trace_file = "RS/sram_read.csv"
):#这里面的num_col都是pe结构的
    outfile = open(sram_read_trace_file,'a')
 
    # There is no data from the left side till the weights are fed in
    # This prefix is to mark the blanks
    prefix  = ""
    for r in range(num_rows):
        prefix += ", "
    ##print("逗号个数，标记空格的前缀prefix",num_rows,prefix)
    # Calculate the convolution window size卷积窗口大小
    r2c = filt_h * filt_w * num_channels #卷积核展开成的一条

    rem = filters_this_fold                 # Track the number of filters yet to process，没有处理结束的过滤器

    #For each wrap around
    for w in range(parallel_window):
        # Number of active columns in this wrap
        cols = min(num_cols, rem)#一次处理col个过滤器
        rem -= cols

        # For each row in the window
        for r in range(r2c):
            entry = str(cycle) + ", " + prefix
            cycle += 1
            
            # In each cycle, for each column feed one weight
            for c in range(cols):
                indx  = w * num_cols + c
                entry += str(col_addr[indx]) + ", "         
                col_addr[indx] += 1

            if cols < num_cols:
                for _ in range(c, num_cols):
                    entry += ", "
            #print("每一行的：entry",entry)
            entry += "\n"
            outfile.write(entry)
 
    outfile.close()
    return cycle


def gen_ifmap_trace(
        cycle = 0,
        num_rows = 4, num_cols = 4,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3, stride = 1,
        parallel_window = 1,
        sram_read_trace_file = "RS/sram_read.csv"
):
    outfile = open(sram_read_trace_file,'a')
    postfix = ""
    for c in range(num_cols):
        postfix += ", "
    
    E_h = math.floor((ifmap_h - filt_h + stride) / stride)
    E_w = math.floor((ifmap_w - filt_w + stride) / stride)
    e2  = E_h * E_w
    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels

    idle = num_rows - (r2c * parallel_window)
    idle = max(idle, 0)
    used_rows = num_rows - idle

    # Adding entries for columns and empty rows
    #print("Idle lanes = " + str(idle))
    idle += num_cols
    for i in range(idle):
        postfix += ", "
    postfix += "\n"

    base_addr = 0
    
    for e in range(int(e2)):
        entry = str(cycle) + ", "
        cycle += 1    

        #print("Cycle= " + str(cycle))
        #Inner loop for all the rows in array
        num_rows = r2c 
        row_entry = []
        for r in range(num_rows):
            row_idx = math.floor(r / rc)  # math.floor to get in integral value
            col_idx = r % rc 
            add = base_addr + row_idx * hc + col_idx 
            #print("Row idx " + str(row_idx) + " col_idx " + str(col_idx) +" add " + str(add))
            row_entry.append(add)

        # Reverse the printing order
        # Reversal is needed because the filter are stored in upside down order in the array
        # ie. last row has the first weight element
        l = len(row_entry)
        #print("Parallel windows = " + str(parallel_window))
        for w in range(parallel_window):
            #print("Window = " + str(w))
            for ridx in range(l):
                entry += str(row_entry[l - ridx -1]) + ", "

        entry += postfix
        outfile.write(entry)

        # Calculate the IFMAP addresses for next cycle
        px_this_row = (e+1) % E_w
        if px_this_row == 0:
            #print("New row")
            ifmap_row = math.floor(base_addr / hc)
            base_addr = (ifmap_row +  stride) * hc
        else:
            base_addr += stride * num_channels
        #print("OFAMP px = " + str(e+1) + " base_addr: " + str(base_addr))

    outfile.close()
    return cycle, used_rows


def gen_trace_filter_partial(
                    col_addrs=[],       #Ensure that this takes care of the v_folding
                    cycle=0,
                    num_rows=4,
                    remaining=4,
                    sram_read_trace_file="RS/sram_read.csv"
):
        outfile = open(sram_read_trace_file, 'a')
        num_cols = len(col_addrs)

        # output formatting: Add empty commas for row addresses as no element is fed from the left
        prefix = ""
        for r in range(num_rows):
            prefix += ", "

        # Entries per cycle 
        for r in range(remaining):              # number of rows this cycle
            entry = str(cycle) + ", " + prefix

            for c in range(num_cols):
                entry += str(col_addrs[c]) + ", "
                col_addrs[c] += 1
            
            cycle += 1
            entry += "\n"
            outfile.write(entry)

        outfile.close()

        return cycle, col_addrs 


def gen_trace_ifmap_partial(
                    cycle = 0,
                    num_rows = 4, num_cols = 4,
                    remaining=4,
                    num_filters = 8,            #   
                    remaining_filters = 0,      # These two are used to track the reads of PS
                    ifmap_h = 4, ifmap_w = 4,
                    filt_h = 3, filt_w = 3,
                    num_channels = 3,
                    stride = 1, 
                    ifmap_base = 0, ofmap_base = 2000000,
                    sram_read_trace_file = "RS/sram_read.csv"
):
    outfile = open(sram_read_trace_file, 'a')
    postfix = ""
    for c in range(num_cols):
        postfix += ", "
    postfix += "\n"

    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels
    E_w = (ifmap_w - filt_w + stride) / stride 
    E_h = (ifmap_h - filt_h + stride) / stride 

    num_ofmap_px = E_h * E_w
    index = r2c - remaining
    base_addr = 0 
            
    filter_done = num_filters - remaining_filters
    #outfile.write(str(filter_done) + ", " + str(num_filters)+", "+str(remaining_filters)+", "+ "\n")
    #ofmap_offset = filter_done * num_ofmap_px
    ofmap_offset = filter_done
    effective_cols = min(remaining_filters, num_cols)
    tick = 0                                # Proxy for clock to track input skewing

    # Outerloop for all ofmap pixels in an ofmap channel
    for e in range(int(num_ofmap_px)):
        entry = str(cycle) + ", "
        cycle += 1    

        #print("Cycle= " + str(cycle))
        #Inner loop for all the rows in array
        num_rows = min(num_rows, remaining)
        row_entry = []
        for r in range(num_rows):
            row_idx = math.floor((index+r) / rc)  # math.floor to get in integral value
            col_idx = (index+r) % rc 
            add = base_addr + row_idx * hc + col_idx 
            #print("Row idx " + str(row_idx) + " col_idx " + str(col_idx) +" add " + str(add))
            row_entry.append(add)

        # Reverse the printing order
        # Reversal is needed because the filter are stored in upside down order in the array
        # ie. last row has the first weight element
        l = len(row_entry)
        for ridx in range(l):
            entry += str(row_entry[l - ridx -1]) + ", "

        # In case of partial mapping
        # index > 0 implies that there is a partial sum generated from prev h_fold
        # This partial sum is now fed from the top to be summed with the PS generated in this h_fold
        # The following part print the read addresses for PS
        # Anand : TODO, Implementation choice, do not support right now
        '''
        if index > 0:
            postfix = ""
            for c in range(effective_cols):
                if (tick - c) > -1:                       # Track PS reads for skew
                    a = (e - c) * num_filters + c        # e - c: Taking care of skew by c cycles
                    a = a + ofmap_base + ofmap_offset
                    postfix += str(a) + ", "
                else:
                    postfix += ", "
            tick += 1
            #print("Tick =", str(tick) + "Postfix= " + postfix)
            postfix += "\n"
        '''
        entry += postfix
        outfile.write(entry)

        px_this_row = (e+1) % E_w
        if px_this_row == 0:
            #print("New row")
            ifmap_row = math.floor(base_addr / hc)
            base_addr = (ifmap_row + stride) * hc
        else:
            base_addr += stride * num_channels
        #print("OFAMP px = " + str(e+1) + " base_addr: " + str(base_addr))

    outfile.close()
    return cycle


def gen_trace_ofmap(
                    cycle = 0,
                    num_rows = 4, num_cols =4,
                    ofmap_base = 2000000,
                    parallel_window = 1,
                    window_size = 27,
                    num_ofmap_px = 16,      # This is per ofmap channel
                    filters_done = 0,       # To track v fold
                    num_filter   = 8,       # To track if all filters have finished
                    sram_write_trace_file = "RS/sram_write.csv"
):
    outfile = open(sram_write_trace_file,'a')
    #cycle = num_cols + cycle     # Accounts for the time taken to reduce accross all cols

    # Corner case when parallel_window = 1, but num_filter < num_cols
    if parallel_window > 1:
        cycle += num_cols
        cycle += window_size                # window_size == r2c
    else:
        rem    = (num_filter - filters_done)
        cycle += min(rem, num_cols)
        cycle += window_size

    #ofmap_add_offset  = filters_done * num_ofmap_px
    ofmap_add_offset  = filters_done
    remaining_filters = num_filter - filters_done
    
    effective_cols    = num_cols * parallel_window
    effective_cols    = min(effective_cols, remaining_filters)

    for e in range(int(num_ofmap_px)):
        entry = str(cycle) + ", "
        cycle += 1
        
        done = filters_done
        for col in range(effective_cols):
            if done < num_filter:
                a = e * num_filter + col                # z first row major
                a = a + ofmap_add_offset + ofmap_base
                entry += str(a) + ", "
            else: 
                # Code should not enter this part
                entry += "!, "

        entry += "\n"
        outfile.write(entry)

    outfile.close()
    return cycle


# Trace generation for moving generated ofmap data in cases when only partial window fits
# This implementation prints out the ofmap pixel in the exact cycle it is generated
# Not used in scale sim at the moment. 
# SCALE sim waits till all the columns finish generating OFMAP.
def gen_trace_ofmap_partial_imm(
                        cycle = 0,
                        num_rows = 4, num_cols =4,
                        ofmap_base = 2000000,
                        num_ofmap_px = 16,
                        num_filter = 8,
                        filters_done = 0,
                        sram_write_trace_file = "RS/sram_write.csv"
):
    outfile = open(sram_write_trace_file,'a')
    start_cycle = num_rows + cycle

    col_addr = []
    for col in range(int(num_cols)):
        a = (filters_done + col)
        col_addr.append(a)
    
    for tick in range(int(num_ofmap_px + num_cols)):
        cycle = start_cycle + tick

        entry = str(cycle) + ", "
        for col in range(int(num_cols)):
            # Condition to maintain skew
            if tick >= col and (tick - col)< num_ofmap_px:
                entry += str(col_addr[col]) + ", "
                col_addr[col] += num_filter
            else:
                entry += ", "
        
        entry += "\n"
        outfile.write(entry)

    outfile.close()


if __name__ == "__main__":
    #value=[226,226, 3,3, 3, 1, 64, 27,64]####这里不是224
    #value=[5,5, 3,3,3, 1, 3, 16,16]
    #value = [5, 5, 3, 3, 3, 1, 20, 8, 8]
    #value = [10, 10, 4, 4, 5, 1, 5, 80, 32]
    #value=[28,28,3,3,1,1,32,256,256]###0
    #value=[26,26,3,3,32,1,64,64,64]###1
    #value=[1,1,1,1,9216,1,128,32,32]###5
    value = [1, 1, 1, 1, 9216, 1, 128, 512, 512]
    #value = [1, 1, 1, 1, 128, 1, 10, 32, 32]###7
    #信息分别是：输入的高；宽； 卷积核高；宽  通道数； 卷积步长； 卷积核个数； 然后是pe尺寸高；宽
    h_h = value[0]
    h_w = value[1]
    r_h = value[2]#2
    r_w = value[3]#2

    c = value[4]

    stru =value[5]

    num = value[6]#9

    dim_h = value[7]
    dim_v = value[8]

    sram_traffic(
        dimension_rows = dim_h,
        dimension_cols = dim_v,

        ifmap_h = h_h, ifmap_w = h_w,
        filt_h = r_h, filt_w = r_w, 
        num_channels = c,
        strides = stru,

        num_filt = num
    )
