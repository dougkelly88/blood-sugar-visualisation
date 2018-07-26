import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import math
import scipy.ndimage.measurements as im_meas
from mpl_toolkits.axes_grid1 import make_axes_locatable

def minus_time(t1, t2):
    a = datetime.timedelta(hours=t1.hour, minutes=t1.minute, seconds=t1.second)
    b = datetime.timedelta(hours=t2.hour, minutes=t2.minute, seconds=t2.second)
    return a - b

def plus_time(t1, t2):
    a = datetime.timedelta(hours=t1.hour, minutes=t1.minute, seconds=t1.second)
    b = datetime.timedelta(hours=t2.hour, minutes=t2.minute, seconds=t2.second)
    return a + b
    
def mealtime(time_in, variability_in_mins):
    mu = datetime.timedelta(hours=time_in.hour, minutes=time_in.minute).total_seconds()
    sigma = variability_in_mins * 60 / 5
    meal_time_s = np.random.normal(mu, sigma)
    h = int(math.floor(meal_time_s / (60 * 60)))
    m = int(math.floor((meal_time_s - h * 60 * 60)/60))
    s = int(math.floor(meal_time_s - h * 60 * 60 - m * 60))
    if (h > 23):
        h = 23;
        m = 59;
        s = 59;
    meal_time = datetime.time(hour=h, minute=m, second=s)
    return meal_time
    
def add_daily_scatter(df, date_to_plot, ax):
    day_df = df.loc[df['date'] == date_to_plot]
    meal_times = np.asarray([datetime.time(hour=8, minute=0), 
                  datetime.time(hour=12, minute=30), 
                  datetime.time(hour=18, minute=30), 
                  datetime.time(hour=23, minute=0)])
    meal_variability_mins = np.asarray([30, 60, 90, 120])
    latest_available_time = day_df['time'].iloc[0]
    meal_variability_mins = meal_variability_mins[meal_times < latest_available_time].tolist()
    meal_times = meal_times[meal_times < latest_available_time].tolist()
    
    ts = [mealtime(t, r) for (t, r) in zip(meal_times, meal_variability_mins)]
    closest_ts = [day_df['time'][((day_df['time'].apply(lambda x: minus_time(x,t))).abs().argsort()[:1])] for t in ts]
    mus = [day_df.loc[day_df['time'] == t]['BG, mmoll-1'] for t in closest_ts]
    readings = [np.around(np.random.normal(mu, 0.05)) for mu in mus]
    ax.scatter(closest_ts, readings)
    

def plot_daily_BG(df, date_to_plot, ax):
    ax.axhspan(2.0, 3.5, alpha=0.25, color='r')
    ax.axhspan(12.0, 18.0, alpha=0.25, color='r')
    ax.axhspan(8.0, 12.0, alpha=0.25, color='orange')
    df.loc[df['date'] == date_to_plot].set_index(df.loc[df['date'] == date_to_plot]['time'])['BG, mmoll-1'].plot(ax=ax)
    ax.set_ylabel('BG, mmoll-1')
    ax.set_xlabel('Time')
    ax.set_ylim((2.0, 18.0))
    ax.set_xlim((datetime.time(hour=0), datetime.time(hour=23, minute=59)))
    t = [datetime.time(hour=4*x) for x in range(6)]
    t.append(datetime.time(hour=23, minute=59))
    ax.set_xticks(t)
    add_daily_scatter(df, date_to_plot,  ax)
    
def lastWday(adate, w):
    MON, TUE, WED, THU, FRI, SAT, SUN = range(7)
    """Mon:w=0, Sun:w=6"""
    delta = (adate.weekday() + 6 - w) % 7 + 1
    return adate - datetime.timedelta(days=delta)
    
def percentageTimeInTarget(df, startdate, enddate):
    time_bands = {'Overnight': (0, 7), 
                 'Pre-breakfast': (7,8), 
                 'Pre-lunch': (10.5, 12), 
                 'Pre-dinner': (16, 18)}
    bg_bands = {'Overnight': (5.0, 9.0), 
                 'Pre-breakfast': (4.0, 8.0), 
                 'Pre-lunch': (3.5, 8.0), 
                 'Pre-dinner': (3.5, 8.0)}
    output = {}
    for key, value in time_bands.iteritems():
        time_bands[key] = (datetime.time(hour=int(math.floor(value[0])), minute=int(value[0]-math.floor(value[0]))),datetime.time(hour=int(math.floor(value[1])), minute=int(value[1]-math.floor(value[1]))))
    sample_df = df.loc[(df['date'] > startdate) & (df['date'] < enddate)]
    #for key in time_bands.iterkeys():
    #    output[key] = 
    
def roundTimeToLastXMinutes(tm, X):
    tm = datetime.datetime.combine(datetime.date.today(), tm)
    tm = tm - datetime.timedelta(minutes=tm.minute % X,
                             seconds=tm.second,
                             microseconds=tm.microsecond)
    tm = datetime.time(hour=tm.hour, minute=tm.minute)
    return tm

def rollingFilterTimeSeries(series_groupby, filt_half_size_mins, filter_type, q=0.5):
    """ Apply a 2 * filt_half_size_mins minute median filter to a pandas series indexed by time. 
    Boundaries are extended by repeating endpoints"""
    if filter_type == 'mean':
        x = series_groupby.mean()
    elif filter_type == 'median':
        x = series_groupby.median()
    elif filter_type == 'quantile':
        x = series_groupby.quantile(q)
    y = x.copy()
    for idx in range(len(x)):
        low_tidx = (datetime.datetime.combine(datetime.date.today(), x.index[idx]) - datetime.timedelta(minutes=filt_half_size_mins)).time()
        high_tidx = (datetime.datetime.combine(datetime.date.today(), x.index[idx]) + datetime.timedelta(minutes=filt_half_size_mins)).time()
        if (low_tidx < high_tidx):
            if filter_type == 'median':
                y.iloc[idx] = x[(x.index >= low_tidx) & (x.index <= high_tidx)].median()
            elif filter_type == "mean":
                y.iloc[idx] = x[(x.index >= low_tidx) & (x.index <= high_tidx)].mean()
            elif filter_type == "quantile":
                y.iloc[idx] = x[(x.index >= low_tidx) & (x.index <= high_tidx)].quantile(q)
        else:
            if filter_type == 'median':
                y.iloc[idx] = x[(x.index >= low_tidx) | (x.index <= high_tidx)].median()
            elif filter_type == 'mean':
                y.iloc[idx] = x[(x.index >= low_tidx) | (x.index <= high_tidx)].mean()
            elif filter_type == "quantile":
                y.iloc[idx] = x[(x.index >= low_tidx) | (x.index <= high_tidx)].quantile(q)
    return y
	
	
def test(instring):
	print("test OK")
	print(instring)
	
	
def plot_long_term_BG(df, startdate, enddate):
	sample_df = df.loc[(df['date'] > startdate) & (df['date'] < enddate)]
	groupbytime = sample_df.groupby('time')['BG, mmoll-1']
	fig, ax = plt.subplots(1,1)
	ax.axhspan(2.0, 3.5, alpha=0.25, color='r')
	ax.axhspan(12.0, 18.0, alpha=0.25, color='r')
	ax.axhspan(8.0, 12.0, alpha=0.25, color='orange')
	bgplt = rollingFilterTimeSeries(groupbytime, 60, 'median').plot(ax=ax, label='Median BG')
	filt75 = rollingFilterTimeSeries(groupbytime, 60, 'quantile', q=0.75)
	filt25 = rollingFilterTimeSeries(groupbytime, 60, 'quantile', q=0.25)

	ax.fill_between(filt75.index, filt25, filt75, alpha=0.25, color='gray', label='25-75% quantile')
	ax.legend()

	ax.set_ylabel('BG, mmoll-1')
	ax.set_xlabel('Time')
	ax.set_ylim((2.0, 18.0))
	ax.set_xlim((datetime.time(hour=0), datetime.time(hour=23, minute=59)))
	t = [datetime.time(hour=4*x) for x in range(6)]
	t.append(datetime.time(hour=23, minute=59))
	ax.set_xticks(t)
	ax.set_title(startdate.strftime("Median BG in period %b %d, %Y - " + enddate.strftime("%b %d, %Y"))) 
	return ax
	
def percentageTimeInTarget(df, startdate, enddate):
    time_bands = {'Overnight': (0, 7), 
                 'Pre-breakfast': (7,8), 
                 'Pre-lunch': (10.5, 12), 
                 'Pre-dinner': (16, 18)}
    bg_bands = {'Overnight': (5.0, 9.0), 
                 'Pre-breakfast': (4.0, 8.0), 
                 'Pre-lunch': (3.5, 8.0), 
                 'Pre-dinner': (3.5, 8.0)}
    sample_df = df.loc[(df['date'] >= startdate) & (df['date'] <= enddate)]
    output = {}
    
    # for plotting...
    abovelims = []
    inlims = []
    belowlims = []
    
    for key, value in time_bands.iteritems():

        band_start_t = datetime.time(hour=int(math.floor(value[0])), minute=int(value[0]-math.floor(value[0])))
        band_end_t = datetime.time(hour=int(math.floor(value[1])), minute=int(value[1]-math.floor(value[1])))
        sub_sample_df = sample_df.loc[(sample_df['time'] >= band_start_t) & (sample_df['time'] <= band_end_t)]['BG, mmoll-1']
        
        t = sub_sample_df.index
        r = pd.date_range(t.min(), t.max(), freq='30S')
        interp_df = sub_sample_df.reindex(t.union(r)).interpolate('index')
        
        pc_below_target = np.round(100.0 * (interp_df < bg_bands[key][0]).sum() / interp_df.count(),1)
        pc_in_target = np.round(100.0 * ((interp_df >= bg_bands[key][0]) & (interp_df <= bg_bands[key][1])).sum() / interp_df.count(),1)
        pc_above_target = np.round(100.0 * (interp_df > bg_bands[key][1]).sum() / interp_df.count(),1)
        output[key] = (pc_below_target, pc_in_target, pc_above_target)
        
        abovelims.append(pc_above_target)
        belowlims.append(pc_below_target)
        inlims.append(pc_in_target)
    
    # handle plotting
    fig, ax = plt.subplots(1,1)
    idx = len(time_bands) - np.arange(len(time_bands))
    bar_w = 0.35
    rects = []
    rects.append(ax.bar(idx, belowlims, bar_w, color='orange', label='Below target'))
    rects.append(ax.bar(idx, inlims, bar_w, color='g', bottom=belowlims, label='Within target'))
    rects.append(ax.bar(idx, abovelims, bar_w, color='r', bottom=[sum(x) for x in zip(belowlims, inlims)], label='Above target'))
    ax.set_xticks(idx)
    ax.set_xticklabels((bg_bands.keys()))
    ax.set_ylabel("Percentage time in each range")
    ax.set_xlabel("Periods of interest")
    ax.legend()
    ax.set_title(startdate.strftime("Time spent within target ranges in period %b %d, %Y - " + enddate.strftime("%b %d, %Y")))
    
    # handle labeling
    labels = []
    for cidx, container in enumerate(rects):
        for ridx, rect in enumerate(container):
            if (rect.get_height() > 3):
                yloc = rect.get_y() + rect.get_height()/2.0
                xloc = rect.get_x() + rect.get_width()/2.0

                pc_str = str.format("{0} %", output[bg_bands.keys()[ridx]][cidx])
                label = ax.text(xloc, yloc, pc_str, horizontalalignment='center',
                             verticalalignment='center', color='w', weight='bold',
                             clip_on=True)
                labels.append(label)
    
    return output

def plot_hypos(df, startdate, enddate):
	#enddate = datetime.date.today()
	#startdate = datetime.date.today() - datetime.timedelta(days=31)
	sample_df = df.loc[(df['date'] >= startdate) & (df['date'] <= enddate)]
			
	t = sample_df.index
	r = pd.date_range(t.min(), t.max(), freq='30S')
	interp_df = sample_df.reindex(t.union(r)).interpolate('index')
	
	mask = (interp_df['BG, mmoll-1'] < 3.5).values
	lbl, nfeat = im_meas.label((mask).astype(int))
	o = im_meas.find_objects(lbl)
	hypo_starts_t = np.asarray([interp_df.index[x[0].start].time() for x in o])
	
	days = interp_df['date'].unique()
	days = np.unique(interp_df.index.date)
	days = days[1:-1]
	out = np.zeros((len(days), (2*60*24)))
	hypo_starts_dt = [interp_df.index[x[0].start] for x in o]

	for didx, day in enumerate(days):
		daily_BG = interp_df.loc[interp_df.index.date == day]['BG, mmoll-1']
		out[didx, :] = daily_BG
		
	out = np.ma.masked_where(out>3.5, out)
	fig, ax = plt.subplots(1,1)
	cbdum = ax.imshow(out, aspect='auto', cmap='Reds_r', clim=(2.0, 3.5))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(cbdum, cax=cax)

	t = [x * 4 * 2 * 60 for x in range(6)]
	d = [x * 4 for x in range((enddate-startdate).days/4)]
	#t.append(2*60*24 - 1)
	ax.set_xticks(t)
	ax.set_yticks(d)
	tlbl = [datetime.time(hour= int(math.floor(tx / (2 * 60))), minute=0) for tx in t]
	dlbl = [x.strftime("%Y-%b-%d") for x in days[d]]
	#nt = datetime.time(hour=23, minute=59)
	#tlbl.append(nt)
	ax.set_xticklabels(tlbl)
	ax.set_xlabel("Time of day")
	ax.set_yticklabels(dlbl)
	ax.set_ylabel("Date")
	ax.set_title(startdate.strftime("Hypoglycaemic episodes in period %b %d, %Y - " + enddate.strftime("%b %d, %Y")))
	cax.set_ylabel("BG, mmoll-1")
	
	return ax