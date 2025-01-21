# coding=utf-8
MNAME="utils.utils"
"""# 
parallal loadin' for faster retrieval
"""
import itertools, time, datetime, multiprocessing, pandas as pd, numpy as np, pickle, gc, os
from multiprocessing.pool import ThreadPool
from threading import Thread
from pathlib import Path
from queue import Queue
from typing import List, Optional, Tuple, Union, Dict, Any


#####################################################################################
#####################################################################################


def date_now(datenow:Union[str,int,float,datetime.datetime]="", fmt="%Y%m%d",
             add_days=0,  add_mins=0, add_hours=0, add_months=0,add_weeks=0,
             timezone_input=None,
             timezone='Asia/Tokyo', fmt_input="%Y-%m-%d",
             force_dayofmonth=-1,   ###  01 first of month
             force_dayofweek=-1,
             force_hourofday=-1,
             force_minofhour=-1,
             returnval='str,int,datetime/unix'):
    """ One liner for date Formatter
    Doc::

        datenow: 2012-02-12  or ""  emptry string for today's date.
        fmt:     output format # "%Y-%m-%d %H:%M:%S %Z%z"
        date_now(timezone='Asia/Tokyo')    -->  "20200519"   ## Today date in YYYMMDD
        date_now(timezone='Asia/Tokyo', fmt='%Y-%m-%d')    -->  "2020-05-19"
        date_now('2021-10-05',fmt='%Y%m%d', add_days=-5, returnval='int')    -->  20211001
        date_now(20211005, fmt='%Y-%m-%d', fmt_input='%Y%m%d', returnval='str')    -->  '2021-10-05'
        date_now(20211005,  fmt_input='%Y%m%d', returnval='unix')    -->

        date_now(' 21 dec 1012',  fmt_input='auto', returnval='unix')    -->

         integer, where Monday is 0 and Sunday is 6.


        date_now(1634324632848, fmt='%Y-%m-%d', fmt_input='%Y%m%d', returnval='str')    -->  '2021-10-05'

    """
    from pytz import timezone as tzone
    import datetime, time

    if timezone_input is None:
        timezone_input = timezone

    sdt = str(datenow)

    #if isinstance(datenow, int) and len(str(datenow)) == 13:
    #    # convert timestamp from miliseconds to seconds
    #    datenow = datenow / 1000


    if isinstance(datenow, datetime.datetime):
        now_utc = datenow

    elif (isinstance(datenow, float) or isinstance(datenow, int))  and  datenow > 1600100100 and str(datenow)[0] == "1"  :  ### Unix time stamp
        ## unix seconds in UTC
        # fromtimestamp give you the date and time in local time
        # utcfromtimestamp gives you the date and time in UTC.
        #  int(time.time()) - date_now( int(time.time()), returnval='unix', timezone='utc') == 0
        now_utc = datetime.datetime.fromtimestamp(datenow, tz=tzone("UTC") )   ##

    elif len(sdt) > 7 and fmt_input != 'auto':  ## date in string
        now_utc = datetime.datetime.strptime(sdt, fmt_input)

    elif fmt_input == 'auto' and sdt is not None:  ## dateparser
        import dateparser
        now_utc = dateparser.parse(sdt)

    else:
        now_utc = datetime.datetime.now(tzone('UTC'))  # Current time in UTC
        # now_new = now_utc.astimezone(tzone(timezone))  if timezone != 'utc' else  now_utc.astimezone(tzone('UTC'))
        # now_new = now_utc.astimezone(tzone('UTC'))  if timezone in {'utc', 'UTC'} else now_utc.astimezone(tzone(timezone))

    if now_utc.tzinfo == None:
        now_utc = tzone(timezone_input).localize(now_utc)

    now_new = now_utc if timezone in {'utc', 'UTC'} else now_utc.astimezone(tzone(timezone))

    ####  Add months
    now_new = now_new + datetime.timedelta(days=add_days + 7*add_weeks, hours=add_hours, minutes=add_mins,)
    if add_months!=0 :
        from dateutil.relativedelta import relativedelta
        now_new = now_new + relativedelta(months=add_months)


    #### Force dates
    if force_dayofmonth >0 :

        if force_dayofmonth == 31:
            mth = now_new.month 
            if   mth in {1,3,5,7,8,10,12} : day = 31
            elif mth in {4,6,9,11}        : day = 30
            else :
                day = 28             
            now_new = now_new.replace(day=day)

        else :    
            now_new = now_new.replace(day=force_dayofmonth)

    if force_dayofweek >0 :
        actual_day = now_new.weekday()
        days_of_difference = force_dayofweek - actual_day
        now_new = now_new + datetime.timedelta(days=days_of_difference)

    if force_hourofday >0 :
        now_new = now_new.replace(hour=force_hourofday)

    if force_minofhour >0 :
        now_new = now_new.replace(minute=force_minofhour)


    if   returnval == 'datetime': return now_new ### datetime
    elif returnval == 'int':      return int(now_new.strftime(fmt))
    elif returnval == 'unix':     return datetime.datetime.timestamp(now_new)  #time.mktime(now_new.timetuple())
    else:                         return now_new.strftime(fmt)
########################################################################################################
###### File parsing functions ###########################################################################
def glob_glob(dirin="", file_list=[], exclude="", include_only="",
            min_size_mb=0, max_size_mb=500000,
            ndays_past=-1, nmin_past=-1,  start_date='1970-01-02', end_date='2050-01-01',
            nfiles=99999999, verbose=0, npool=1
    ):
    """ Advanced glob.glob filtering.
    Docs::

        dirin:str = "": get the files in path dirin, works when file_list=[]
        file_list: list = []: if file_list works, dirin will not work
        exclude:str  = ""
        include_only:str = ""
        min_size_mb:int = 0
        max_size_mb:int = 500000
        ndays_past:int = 3000
        start_date:str = '1970-01-01'
        end_date:str = '2050-01-01'
        nfiles:int = 99999999
        verbose:int = 0
        npool:int = 1: multithread not working
        
        https://www.twilio.com/blog/working-with-files-asynchronously-in-python-using-aiofiles-and-asyncio
    """
    import glob, copy, datetime as dt, time


    def fun_glob(dirin=dirin, file_list=file_list, exclude=exclude, include_only=include_only,
            min_size_mb=min_size_mb, max_size_mb=max_size_mb,
            ndays_past=ndays_past, nmin_past=nmin_past,  start_date=start_date, end_date=end_date,
            nfiles=nfiles, verbose=verbose,npool=npool):
        """
        Docs::
            dirin:str = "": get the files in path dirin, works when file_list=[]
            file_list: list = []: if file_list works, dirin will not work
            exclude:str  = ""
            include_only:str = ""
            min_size_mb:int = 0
            max_size_mb:int = 500000
            ndays_past:int = 3000
            start_date:str = '1970-01-01'
            end_date:str = '2050-01-01'
            nfiles:int = 99999999
            verbose:int = 0
            npool:int = 1: multithread not working
        """
        if dirin and not file_list:
            files = glob.glob(dirin, recursive=True)
            files = sorted(files)

        if file_list:
            files = file_list

        ####### Exclude/Include  ##################################################
        for xi in exclude.split(","):
            if len(xi) > 0:
                files = [  fi for fi in files if xi not in fi ]

        if include_only:
            tmp_list = [] # add multi files
            for xi in include_only.split(","):
                if len(xi) > 0:
                    tmp_list += [  fi for fi in files if xi in fi ]
            files = sorted(set(tmp_list))

        ####### size filtering  ##################################################
        if min_size_mb != 0 or max_size_mb != 0:
            flist2=[]
            for fi in files[:nfiles]:
                try :
                    if min_size_mb <= os.path.getsize(fi)/1024/1024 <= max_size_mb :   #set file size in Mb
                        flist2.append(fi)
                except : pass
            files = copy.deepcopy(flist2)

        #######  date filtering  ##################################################
        now    = time.time()
        cutoff = 0

        if ndays_past > -1 :
            cutoff = now - ( abs(ndays_past) * 86400)

        if nmin_past > -1 :
            cutoff = cutoff - ( abs(nmin_past) * 60  )

        if cutoff > 0:
            if verbose > 0 :
                print('now',  dt.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                       ',past', dt.datetime.utcfromtimestamp(cutoff).strftime("%Y-%m-%d %H:%M:%S") )
            flist2=[]
            for fi in files[:nfiles]:
                try :
                    t = os.stat( fi)
                    c = t.st_ctime
                    if c < cutoff:             # delete file if older than 10 days
                        flist2.append(fi)
                except : pass
            files = copy.deepcopy(flist2)

        ########################################################################################################
###### File parsing functions ###########################################################################
def glob_glob(dirin="", file_list=[], exclude="", include_only="",
            min_size_mb=0, max_size_mb=500000,
            ndays_past=-1, nmin_past=-1,  start_date='1970-01-02', end_date='2050-01-01',
            nfiles=99999999, verbose=0, npool=1
    ):
    """ Advanced glob.glob filtering.
    Docs::

        dirin:str = "": get the files in path dirin, works when file_list=[]
        file_list: list = []: if file_list works, dirin will not work
        exclude:str  = ""
        include_only:str = ""
        min_size_mb:int = 0
        max_size_mb:int = 500000
        ndays_past:int = 3000
        start_date:str = '1970-01-01'
        end_date:str = '2050-01-01'
        nfiles:int = 99999999
        verbose:int = 0
        npool:int = 1: multithread not working
        
        https://www.twilio.com/blog/working-with-files-asynchronously-in-python-using-aiofiles-and-asyncio
    """
    import glob, copy, datetime as dt, time


    def fun_glob(dirin=dirin, file_list=file_list, exclude=exclude, include_only=include_only,
            min_size_mb=min_size_mb, max_size_mb=max_size_mb,
            ndays_past=ndays_past, nmin_past=nmin_past,  start_date=start_date, end_date=end_date,
            nfiles=nfiles, verbose=verbose,npool=npool):
        """
        Docs::
            dirin:str = "": get the files in path dirin, works when file_list=[]
            file_list: list = []: if file_list works, dirin will not work
            exclude:str  = ""
            include_only:str = ""
            min_size_mb:int = 0
            max_size_mb:int = 500000
            ndays_past:int = 3000
            start_date:str = '1970-01-01'
            end_date:str = '2050-01-01'
            nfiles:int = 99999999
            verbose:int = 0
            npool:int = 1: multithread not working
        """
        if dirin and not file_list:
            files = glob.glob(dirin, recursive=True)
            files = sorted(files)

        if file_list:
            files = file_list

        ####### Exclude/Include  ##################################################
        for xi in exclude.split(","):
            if len(xi) > 0:
                files = [  fi for fi in files if xi not in fi ]

        if include_only:
            tmp_list = [] # add multi files
            for xi in include_only.split(","):
                if len(xi) > 0:
                    tmp_list += [  fi for fi in files if xi in fi ]
            files = sorted(set(tmp_list))

        ####### size filtering  ##################################################
        if min_size_mb != 0 or max_size_mb != 0:
            flist2=[]
            for fi in files[:nfiles]:
                try :
                    if min_size_mb <= os.path.getsize(fi)/1024/1024 <= max_size_mb :   #set file size in Mb
                        flist2.append(fi)
                except : pass
            files = copy.deepcopy(flist2)

        #######  date filtering  ##################################################
        now    = time.time()
        cutoff = 0

        if ndays_past > -1 :
            cutoff = now - ( abs(ndays_past) * 86400)

        if nmin_past > -1 :
            cutoff = cutoff - ( abs(nmin_past) * 60  )

        if cutoff > 0:
            if verbose > 0 :
                print('now',  dt.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                       ',past', dt.datetime.utcfromtimestamp(cutoff).strftime("%Y-%m-%d %H:%M:%S") )
            flist2=[]
            for fi in files[:nfiles]:
                try :
                    t = os.stat( fi)
                    c = t.st_ctime
                    if c < cutoff:             # delete file if older than 10 days
                        flist2.append(fi)
                except : pass
            files = copy.deepcopy(flist2)

        ####### filter files between start_date and end_date  ####################
        if start_date and end_date:
            start_timestamp = time.mktime(time.strptime(str(start_date), "%Y-%m-%d"))
            end_timestamp   = time.mktime(time.strptime(str(end_date), "%Y-%m-%d"))
            flist2=[]
            for fi in files[:nfiles]:
                try:
                    t = os.stat( fi)
                    c = t.st_mtime
                    if start_timestamp <= c <= end_timestamp:
                        flist2.append(fi)
                except: pass
            files = copy.deepcopy(flist2)

        return files

    if npool ==  1:
        return fun_glob(dirin, file_list, exclude, include_only,
            min_size_mb, max_size_mb,
            ndays_past, nmin_past,  start_date, end_date,
            nfiles, verbose,npool)

    else :
        raise Exception('no working with npool>1')
        # from utilmy import parallel as par
        # input_fixed = {'exclude': exclude, 'include_only': include_only,
        #                'npool':1,
        #               }
        # if dirin and not file_list:
        #     fdir = [item for item in os.walk(dirin)] # os.walk(dirin, topdown=False)
        # if file_list:
        #     fdir = file_list
        # res  = par.multithread_run(fun_glob, input_list=fdir, input_fixed= input_fixed,
        #         npool=npool)
        # res  = sum(res) ### merge
        # return res
####### filter files between start_date and end_date  ####################
        if start_date and end_date:
            start_timestamp = time.mktime(time.strptime(str(start_date), "%Y-%m-%d"))
            end_timestamp   = time.mktime(time.strptime(str(end_date), "%Y-%m-%d"))
            flist2=[]
            for fi in files[:nfiles]:
                try:
                    t = os.stat( fi)
                    c = t.st_mtime
                    if start_timestamp <= c <= end_timestamp:
                        flist2.append(fi)
                except: pass
            files = copy.deepcopy(flist2)

        return files

    if npool ==  1:
        return fun_glob(dirin, file_list, exclude, include_only,
            min_size_mb, max_size_mb,
            ndays_past, nmin_past,  start_date, end_date,
            nfiles, verbose,npool)

    else :
        raise Exception('no working with npool>1')
        # from utilmy import parallel as par
        # input_fixed = {'exclude': exclude, 'include_only': include_only,
        #                'npool':1,
        #               }
        # if dirin and not file_list:
        #     fdir = [item for item in os.walk(dirin)] # os.walk(dirin, topdown=False)
        # if file_list:
        #     fdir = file_list
        # res  = par.multithread_run(fun_glob, input_list=fdir, input_fixed= input_fixed,
        #         npool=npool)
        # res  = sum(res) ### merge
        # return res


###################################################################################################
def log(*s, **kw):
    print(*s, flush=True, **kw)


#############################################################################################################
#############################################################################################################
def pd_read_file(path_glob="*.pkl", ignore_index=True,  cols=None, verbose=False, nrows=-1, nfile=1000000, concat_sort=True,
                 n_pool=1, npool=None,
                 drop_duplicates=None, col_filter:str=None,  col_filter_vals:list=None, dtype_reduce=None,
                 fun_apply=None, use_ext=None,   **kw)->pd.DataFrame:
    """  Read file in parallel from disk : very Fast.
    Doc::

        path_glob: list of pattern, or sep by ";"
        :return:
    """
    import glob, gc,  pandas as pd, os

    if isinstance(path_glob, pd.DataFrame ) : return path_glob   ### Helpers

    n_pool = npool if isinstance(npool, int)  else n_pool ## alias
    def log(*s, **kw):  print(*s, flush=True, **kw)
    readers = {
            ".pkl"     : pd.read_pickle, ".parquet" : pd.read_parquet, ".json" : pd.read_json,
            ".tsv"     : pd.read_csv, ".csv"     : pd.read_csv, ".txt"     : pd.read_csv, ".zip"     : pd.read_csv,
            ".gzip"    : pd.read_csv, ".gz"      : pd.read_csv,
     }

    #### File
    if isinstance(path_glob, list):  path_glob = ";".join(path_glob)
    path_glob = path_glob.split(";")
    file_list = []
    for pi in path_glob :
        if "*" in pi : file_list.extend( sorted( glob.glob(pi) ) )
        else :         file_list.append( pi )

    file_list = sorted(list(set(file_list)))
    file_list = file_list[:nfile]
    if verbose: log(file_list)

    ### TODO : use with kewyword arguments ###############
    def fun_async(filei):
            ext  = os.path.splitext(filei)[1]
            if ext is None or ext == '': ext ='.parquet'

            pd_reader_obj = readers.get(ext, None)
            # dfi = pd_reader_obj(filei)
            try :
               dfi = pd_reader_obj(filei, **kw)
            except Exception as e:
               log('Error', filei, e)
               return pd.DataFrame()

            # if dtype_reduce is not None:    dfi = pd_dtype_reduce(dfi, int0 ='int32', float0 = 'float32')
            if col_filter is not None :       dfi = dfi[ dfi[col_filter].isin( col_filter_vals) ]
            if cols is not None :             dfi = dfi[cols]
            if nrows > 0        :             dfi = dfi.iloc[:nrows,:]
            if drop_duplicates is not None  : dfi = dfi.drop_duplicates(drop_duplicates, keep='last')
            if fun_apply is not None  :       dfi = dfi.apply(lambda  x : fun_apply(x), axis=1)
            return dfi



    ### Parallel run #################################
    import concurrent.futures
    dfall  = pd.DataFrame(columns=cols) if cols is not None else pd.DataFrame()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_pool) as executor:
        futures = []
        for i,fi in enumerate(file_list) :
            if verbose : log("file ", i, end=",")
            futures.append( executor.submit(fun_async, fi ))

        for future in concurrent.futures.as_completed(futures):
            try:
                dfi   = future.result()
                dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)
                del dfi; gc.collect()
            except Exception as e:
                log('error', e)
    return dfall

def pd_to_file(df, filei,  check=0, verbose=True, show='shape', index=False, sep="\t",   **kw):
  """function pd_to_file.
  Doc::
          
        Args:
            df:   
            filei:   
            check:   
            verbose:   
            show:   
            **kw:   
        Returns:
            
  """
  import os, gc
  from pathlib import Path
  parent = Path(filei).parent
  os.makedirs(parent, exist_ok=True)
  ext  = os.path.splitext(filei)[1]
  if   ext == ".pkl" :       df.to_pickle(filei,  **kw)
  elif ext == ".parquet" :   df.to_parquet(filei, **kw)
  elif ext in [".csv" ,".txt"] :  
    df.to_csv(filei, index=False, sep=sep, **kw)       

  elif ext in [".json" ] :  
    df.to_json(filei, **kw)       

  else :
      log('No Extension, using parquet')
      df.to_parquet(filei + ".parquet", **kw)

  if verbose in [True, 1] :  log(filei)        
  if show == 'shape':        log(df.shape)
  if show in [1, True] :     log(df)
     
  if check in [1, True, "check"] : log('Exist', os.path.isfile(filei))
  #  os_file_check( filei )

  # elif check =="checkfull" :
  #  os_file_check( filei )
  #  dfi = pd_read_file( filei, n_pool=1)   ### Full integrity
  #  log("#######  Reload Check: ",  filei, "\n"  ,  dfi.tail(3).T)
  #  del dfi; gc.collect()
  gc.collect()



    ###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()