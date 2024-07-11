import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from scipy.stats import f
from statsmodels.tsa.stattools import adfuller

# Constants
def LEVEL():
    '''
    '''
    return 0.95

def L0(algorithm):
    if algorithm == 'chow' or algorithm == 'ADF':
        return 5
    else:
        return 20

def export_data(series, pointers, nseg, show_pic, output_name):
    '''
        Creates a pandas DataFrame with the segmentation result, consisting of 7 columns:
                - Segment: Segment number.
                - Start: First point of the segment.
                - Finish: Last point of the segment.
                - Size: number of points in the segment.
                - Mean: Mean value of the segment.
                - Median: Median of the segment.
                - Sigma: Variance of the segment.
        
        Input:
            - series: original series that was segmented.
            - pointers: array containg the indexes of the starting points of all segments.
            - nseg: number of segments.
            - show_pic: if 1, print the series identifying the segments and save it as .png.
            - output_name: specifies the output name for the picture file.
            
        Output: a pandas DataFrame.
    '''
    # All arrays (columns of the DataFrame) are initialized.
    index = np.arange(1, nseg+1)
    start = np.zeros(nseg)
    finish = np.zeros(nseg)
    means = np.zeros(nseg)
    medians = np.zeros(nseg)
    variance = np.zeros(nseg)
    segment_size = np.zeros(nseg)
    
    segment = series[int(pointers[0])+1 : int(pointers[1])+1]
    segment_size[0] = segment.size
    start[0] = pointers[0]
    finish[0] = pointers[1]
    means[0] = segment.mean()
    variance[0] = segment.var()
    for j in np.arange(1, nseg):
        segment = series[int(pointers[j])+1 : int(pointers[j+1])+1]
        segment_size[j] = segment.size
        start[j] = pointers[j] + 1
        finish[j] = pointers[j+1]
        means[j] = segment.mean()
        medians[j] = np.median(segment)
        variance[j] = segment.var()
        
    result = pd.DataFrame({"index": index,
                           "start": start,
                           "finish": finish,
                           "size": segment_size,
                           "mean": means,
                           "median": medians,
                           "variance": variance})
    
    if show_pic == 1:
        means = np.zeros(0)
        upper_band = np.zeros(0)
        lower_band = np.zeros(0)

        fig = plt.figure()
        fig.set_size_inches(30, 5)
        ax1 = fig.add_subplot(1,1,1)

        time = np.arange(series.size)
        ax1.plot(time, series, color = 'silver')

        for i in result.index:
            size = int(result.loc[i,'finish'] - result.loc[i, 'start'] + 1)
            time = np.arange(result.loc[i, "start"], result.loc[i, "finish"] + 1)
            means = np.full([size], result.loc[i, 'mean'])
            upper_band = result.loc[i, 'mean'] + np.full([size], np.sqrt(result.loc[i, 'variance']))
            lower_band = result.loc[i, 'mean'] - np.full([size], np.sqrt(result.loc[i, 'variance']))
            ax1.plot(time, upper_band, 'r--', linewidth = 1)
            ax1.plot(time, lower_band, 'r--', linewidth = 1)
            ax1.plot(time, means, 'k-', linewidth = 2, drawstyle='steps-post', label='step-post')

        ax1.set_xticks(range(0, int(time[len(time)-1]), 500))
        ax1.set_xlabel('interval number')
        ax1.set_ylabel('value')
        fig.savefig(output_name+".png")
        plt.show()
         
    return result.set_index('index')

### KS-distance ###
@jit    
def kstwo(left_seg, left_seg_size, right_seg, right_seg_size):
    '''
        Calculate the KS-Distance between left_seg and right_seg.
        
        Input:
            - left_seg: a numpy array corresponding to a left_segment of the series.
            - right_seg: a numpy array corresponding to a right_segment of the series.
            - left_seg_size: size of the left segment.
            - right_seg_size: size of the right segment.
    '''
    j1 = 0
    j2 = 0
    fn1 = 0.0
    fn2 = 0.0
    dks = 0.0
    
    left_temp = np.sort(left_seg)
    right_temp = np.sort(right_seg)

    while (j1 < left_seg_size and j2 < right_seg_size):
        d1 = left_temp[j1]
        d2 = right_temp[j2]
        if d1 <= d2:
            fn1 = (j1 + 1)/left_seg_size
            j1 += 1
        if d1 >= d2:
            fn2 = (j2 + 1)/right_seg_size
            j2 += 1
        dks_temp = math.fabs(fn2 - fn1)
        if(dks_temp > dks):
            dks = dks_temp
       
    effective_size = math.sqrt( left_seg_size*right_seg_size / (left_seg_size + right_seg_size) )
   
    return dks*effective_size

    
def dksmax(series, tLf, tRf):
    '''
        Calculate the maximum Kolmogorov-Smirnov Distance within a series.
        The algorithm will run through all points within the series, passing through all possible segmentations
        and calculating the KS-Distance between each segment. 
        
        Input: 
            - series: an numpy array of the data. The maximum KS-Distance will be calculate for the whole data or
            for a subset defined by tLf and tRf.
            - tLf: index of the first point of the series subset that will be analyzed.
            - tRf: index of the last point of the series subset that will be analyzed.
            
        Output: 
            = [dmax, idmax]: a list, where dmax is the maximum KS-Distance and idmax is the index of the segmentation
            point corresponding to dmax.
    '''
    dmax = 0.0      # stores the maximum distance
    idmax = 0       # stores the segmentation index.
    
    for k in range(tLf, tRf):
        left_seg_size = k - tLf + 1
        right_seg_size = tRf - k
    
        left_segment = series[tLf : k+1]        # copies data from series to the left segment: from index tLf to k
        right_segment = series[k+1 : tRf+1]     # copies data from series to the right segment: from index k + 1 to tRf

        # Calculate the KS-Distance between left_segment and right_segment
        d = kstwo(left_segment, left_seg_size, right_segment, right_seg_size)
        
        if d > dmax:
            dmax = d
            idmax = k
        
    return [dmax, idmax]

### Chow-Test ###
def f_value(y1, x1, y2, x2):
    def find_rss (y, x):
        A = np.vstack([x, np.ones(len(x))]).T
        rss = np.linalg.lstsq(A, y, rcond=None)[1]
        length = len(y)
        return (rss, length)

    rss_total, n_total = find_rss(np.append(y1, y2), np.append(x1, x2))
    rss_1, n_1 = find_rss(y1, x1)
    rss_2, n_2 = find_rss(y2, x2)
    
    chow_nom = (rss_total - (rss_1 + rss_2)) / 2
    chow_denom = (rss_1 + rss_2) / (n_1 + n_2 - 4)
    return chow_nom / chow_denom


def p_value(y1, x1, y2, x2, **kwargs):
    F = f_value(y1, x1, y2, x2, **kwargs)
    if not F:
        F = 0
    df1 = 2
    df2 = len(x1) + len(x2) - 4
    if F == 0:
        p_val =1
    else:
        p_val = f.sf(F[0], df1, df2)
    return F, p_val

def chow_test(series, tLf, tRf):
    
    dmax = 0.0      # stores the maximum distance
    idmax = 0       # stores the segmentation index.
    p_ = 0.0
    
    for k in range(tLf, tRf):
        left_seg_size = k - tLf + 1
        right_seg_size = tRf - k
    
        left_segment = series[tLf : k+1]        # copies data from series to the left segment: from index tLf to k
        right_segment = series[k+1 : tRf+1]     # copies data from series to the right segment: from index k + 1 to tRf

        f, p = p_value(y1=left_segment, x1=left_segment, 
                       y2=right_segment, x2=right_segment)
        if f > dmax:
            dmax = f
            idmax = k
            p_ = p
    return [dmax, idmax]

### ADF test ###
def ADF_test(series, tLf, tRf):
    
    dmax = np.inf     # stores the maximum distance
    idmax = 0       # stores the segmentation index.
    
    for k in range(tLf, tRf):
        left_seg_size = k - tLf + 1
        right_seg_size = tRf - k -1
    
        left_segment = series[tLf : k+1]        # copies data from series to the left segment: from index tLf to k
        right_segment = series[k+1 : tRf+1]     # copies data from series to the right segment: from index k + 1 to tRf
        
        if left_seg_size > 5 :
            left_p = adfuller(left_segment)[1]
            #right_p = adfuller(right_segment)[1]
            p = left_p #+ right_p
            if p < dmax:
                dmax = p
                idmax = k
        else:
            pass

    return [dmax, idmax]

def segment(data, show_steps = 0, show_pic = 0, output_name = "segmentation_result", algorithm='KS'):
    '''
        The main function of the segmentation algorithm.
        
        Input: 
            Necessary:
                - data: a numpy array containing the series to be segmented.
            
            Optional:
                - show_steps: 0 as default; if 1, will shows the status of each step in the segmentation process
                - show_pic: 0 as default: if 1, print the series identifying the segments and save it as .png.
                - output_name: "segmentation_result" as default; specifies the output name for the picture file.
                
        Output:
            A pandas DataFrame with the segmentation result, consisting of 6 columns:
                - Segment: Segment number.
                - Start: First point of the segment.
                - Finish: Last point of the segment.
                - Mean: Mean value of the segment.
                - Sigma: Variance of the segment.
    '''
    
    series, seriesSize = np.insert(data, 0, 0, axis = 0), np.size(data)
    max_seg_number = math.ceil(seriesSize / L0(algorithm))
    print("Series Size = ", seriesSize, "\nL0 (Minimum Segment Size) = ", L0(algorithm), "\nMaximum Number of Segments ( ceil(n / L0) ) = ", max_seg_number)
    
    # Definition of necessary arrays.
    # pointers: will countain the index numbers of the starting points of each segment.
    # segIndicator: each index 'i' will indicate if the segment starting at pointers[i] is segmentable or not.
    #   if segIndicator[i] = 1, then it's already exausted, i.e. it cannot be segmented anymore, otherwise, segIndicator = 0.
    # pointers_temp: auxiliary array to save the previous values of the "pointers" array.
    # segIndicator_temp: auxiliary array to save the previous values of the "segIndicator" array.
    pointers = np.zeros(max_seg_number)
    segIndicator = np.zeros(max_seg_number)
    pointers_temp = np.zeros(max_seg_number)
    segIndicator_temp = np.zeros(max_seg_number)
    
    # Initially, the series have only one segment, starting at 0 (p[0] = 0, as defined) and ending at seriesSize - 1,
    # therefore, the starting point of the "second segment" is set as equal to seriesSize.
    # nseg: counts the number of segments detected.
    # step: counts the steps taken ...
    # segmenting: used to end the segmentation loop when it's not possible to find any significant segment, or with size greather than L0.
    pointers[1] = seriesSize
    nseg = 1
    step = 0
    segmenting = 1
    
    while segmenting == 1:
        step += 1
        new_segments = 0 # Is set to 1 if a new segment is detected
        segmenting = 0   # Is set to 1 if a new segment is detected
        
        # The values in the arrays "pointers_temp" and "segIndicator_temp" will be modified along the algorithm
        pointers_temp[0 : nseg+1] = pointers[0 : nseg+1]
        segIndicator_temp[0 : nseg+1] = segIndicator[0 : nseg+1]
        
        if show_steps == 1:
            print("\n------------------------------------------------------")
            print("Step ", step,". Current state:")
            print("> Segment [%d]\t Start: %d\t Non-segmentable: %d" % (1, pointers[0], segIndicator[0]))
            for j in np.arange(1, nseg):
                print("> Segment [%d]\t Start: %d\t Non-segmentable: %d" % (j+1, pointers[j] + 1, segIndicator[j]))
            print("------------------------------------------------------\n")
            
        for j in np.arange(0, nseg):
            if(segIndicator[j] == 0):
                dmax, idmax = 0,0
                if algorithm == 'KS':
                    dmax, idmax = dksmax(series, int(pointers[j]) + 1, int(pointers[j + 1]))
                elif algorithm == 'chow':
                    dmax, idmax = chow_test(series, int(pointers[j]) + 1, int(pointers[j + 1]))
                elif algorithm == 'ADF':
                    dmax, idmax = ADF_test(series, int(pointers[j]) + 1, int(pointers[j + 1]))
                    
                if show_steps == 1:
                    print("Analyzing segment starting at %d and ending at %d"%(pointers[j] + 1, pointers[j+1]))
                    print(">> Maximum KS-Distance = %lg\t Segmentation Index = %d" % (dmax, idmax))
                
                # Verify if the segment size is greater or equal to the minimum size
                is_size_min = (idmax - pointers[j]) >= L0(algorithm) and (pointers[j+1] - idmax) >= L0(algorithm)
                if not is_size_min:
                    segIndicator_temp[j + new_segments] = 1
                    if show_steps == 1:
                        print("Not segmentable: [%d...%d] < L0\n" %(pointers[j] + 1, pointers[j+1]))
                else:
                    # Definition of the Critical Distance 'dcrit'.
                    # The calculated KS-Distance 'dmax' must be at least equal to 'dcrit' so that the
                    # proposed segmentation is actually significant.
                    dcrit = np.inf
                    if algorithm == 'ADF':
                        dcrit = 0.05
                    else:
                        #dcrit = 1.41 * math.exp(0.15*math.log(math.log(pointers[j+1] - pointers[j]) - 1.74)) # 90%
                        dcrit = 1.52 * math.exp(0.14*math.log(math.log(pointers[j+1] - pointers[j]) - 1.80)) # 95%
                        #dcrit = 1.72 * math.exp(0.13*math.log(math.log(pointers[j+1] - pointers[j]) - 1.86)) # 99%
                    
                    # Verify if the segmentation is significant
                    if algorithm == 'ADF':
                        is_significant = dmax < dcrit
                    else:
                        is_significant = dmax > dcrit
                    if not is_significant:
                        segIndicator_temp[j + new_segments] = 1
                        if show_steps == 1:
                            print("Not significant: dmax(%lg) < dcrit(%lg)\n"%(dmax, dcrit))
                    else:
                        # The proposed segmentation is significant and the segment size is at least L0.
                        segmenting = 1  
                        new_segments += 1
                        
                        if(j + new_segments > max_seg_number):
                            print("ERROR\n")
                        
                        pointers_temp = np.insert(pointers_temp, j + new_segments, idmax)
                        segIndicator_temp = np.insert(segIndicator_temp, j + new_segments, 0)
                        
                        flag1 = pointers_temp[j + new_segments + 1] - pointers_temp[j + new_segments] >= 2 * L0(algorithm)
                        
                        if not flag1:
                            segIndicator_temp[j + new_segments] = 1
                        else:
                            if show_steps == 1:
                                print("Accepted new segment - Starting Index: %d\t Segmentation Point : %d\t Ending index: %d\n"%(pointers[j], idmax, pointers[j+1]))
                            
        nseg += new_segments
        pointers[0:nseg+1] = pointers_temp[0 : nseg+1]
        segIndicator[0 : nseg+1] = segIndicator_temp[0 : nseg+1]
        
    
    print("\n\nFinished.")
    pointers[nseg] = pointers[nseg] - 1
    print("> Segment [%d]\t Start: %d\t Non-segmentable: %d" % (1, pointers[0], segIndicator[0]))  
    for j in np.arange(1, nseg):
            print("> Segment [%d]\t Start: %d\t Finish: %d\t Non-segmentable: %d" % (j+1, pointers[j] + 1, pointers[j+1], segIndicator[j]))
    
    return export_data(series, pointers, nseg, show_pic, output_name)