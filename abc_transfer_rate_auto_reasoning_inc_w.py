#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
import os
import sys
os.environ['GROUP_ID'] = 'g_sng_im_sng_imappdev_edu'
os.environ['GAIA_ID'] = '1947'
#spark = SparkSession.builder.config('spark.driver.memory', '16g').config('spark.executor.cores',8).config('spark.executor.memory', '16g').getOrCreate()
spark = SparkSession.builder.config('spark.driver.memory', '16g').config('spark.executor.cores',4).config('spark.tdw.meta.forceRequest','true').config('spark.executor.memory', '16g').getOrCreate()
from pytoolkit import TDWSQLProvider
provider = TDWSQLProvider(spark, db='sng_imappdev_edu_app')
# 获取时间：us上时间参数设置举例：{YYYYMM} {YYYYMMDD} {YYYYMMDDHH}
YYYYMMDD=sys.argv[1]


#读取数据
table_name='abc_dwd_sale_transfer_rate_reasoning_inc_w'
sql_content='select ftime,member_id,school_subject,lesson_type,start_dayno,\
lesson_level,camp_id,chan_name,sku,buy_camp_money,transfer_money,is_transfer,learning_detail,\
add_friend,add_group,teacher_id,person_id,teacher_name,teacher_location,buy_order_id,adtag,\
channel_level_1,channel_level_2,channel_level_3,channel_level_4,adv_channel_level_4,teacher_priority,\
father_depart_id,depart_id from '+table_name+' where ftime='+YYYYMMDD
print(sql_content)
provider.table(table_name).createOrReplaceTempView(table_name)
res=spark.sql(sql_content).collect()

import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import itertools
pd.set_option('display.max_columns', 100)

raw_df=pd.DataFrame(res)
raw_df.columns=['ftime','member_id','school_subject','lesson_type','start_dayno',\
'lesson_level','camp_id','chan_name','sku','buy_camp_money','transfer_money','is_transfer','learning_detail',\
'add_friend','add_group','teacher_id','person_id','teacher_name','teacher_location','buy_order_id','adtag',\
'channel_level_1','channel_level_2','channel_level_3','channel_level_4','adv_channel_level_4','teacher_priority','father_depart_id','depart_id']
#维度名称映射字典
##一级渠道
channel_level_1_name_dict={'adv':'广告投放','out':'合作渠道','reco':'转介绍','nature':'自然流量','empty':'空渠道',
                           'ad':'品牌广告','adshare':'自然分享','undefined':'未识别渠道','unknown':'未知渠道',
                           'fx':'分销','in':'内部渠道'}
channel_level_2_name_dict={'dy':'抖音','out':'合作渠道','gdt':'广点通','wx':'微信大投放','teacher':'老师转化',
                           'wx-old':'微信','xcx':'小程序','app':'内容锁','zrllgzh':'公众号','activity':'活动','other':'其他',
                           'gzhpush':'公众号推送','empty':'空渠道','kol':'KOL','campaign':'品牌广告','adshare':'自然分享',
                           'bd':'百度','home':'首页','sharexcx':'小程序分享','outbj':'马老师渠道','baidu':'百度',
                           'campany':'公司合作'}
teacher_location_name={'sz_employee':'深圳','cq_employee':'重庆','shouhou':'售后','wh_employee':'武汉'}

raw_df['teacher_location']=raw_df['teacher_location'].apply(lambda x: teacher_location_name.get(x,x))
#处理维度空值
raw_df['teacher_priority'].replace('\\N',-1,inplace=True)
raw_df['father_depart_id'].replace('\\N',-1,inplace=True)
raw_df['depart_id'].replace('\\N',-1,inplace=True)
raw_df['sku'].replace('\\N',-1,inplace=True)
#广告渠道账号信息放入3级渠道
raw_df.loc[raw_df['channel_level_1']=='adv','channel_level_3']=raw_df.loc[raw_df['channel_level_1']=='adv','adv_channel_level_4']
#增加用户计数
raw_df['user_cnt']=1
raw_df['lesson_weeks']=3
#增加节数
raw_df.loc[raw_df['lesson_type']=='train','lesson_weeks']=1
raw_df.loc[raw_df['lesson_level']=='s110','lesson_weeks']=2
raw_df.loc[raw_df['lesson_level']=='s210','lesson_weeks']=2
#课程级别标准化
raw_df.loc[raw_df['lesson_level']=='s110','lesson_level']='s1'
raw_df.loc[raw_df['lesson_level']=='s210','lesson_level']='s2'
#枚举科目和课程类型，用于分别计算结果
school_subject_list=list(set(raw_df['school_subject'].tolist()))
lesson_type_list=list(set(raw_df['lesson_type'].tolist()))
raw_df['buy_camp_money']=raw_df['buy_camp_money'].astype(str)
print(raw_df.shape)
#期次排序字典
order_camp_dict_all={}
camp_order_dict_all={}
for s in school_subject_list:
    for t in lesson_type_list:
        tmp_df=raw_df[(raw_df['school_subject']==s)&(raw_df['lesson_type']==t)]
        camp_order=pd.DataFrame(tmp_df['start_dayno'].drop_duplicates())
        camp_order.index=camp_order['start_dayno']
        camp_order['camp_order']=camp_order['start_dayno'].rank(method='min')
        camp_order_dict=camp_order['camp_order'].to_dict()
        camp_order_dict_all[(s,t)]=camp_order_dict
        camp_order.index=camp_order['camp_order']
        order_camp_dict=camp_order['start_dayno'].to_dict()
        order_camp_dict_all[(s,t)]=order_camp_dict
print(order_camp_dict_all)
print(camp_order_dict_all)
#计算指标
cal_metrics_list=['camp_user_cnt','transfer_user_cnt','camp_user_ratio','transfer_user_ratio','transfer_rate']
#归因逻辑树，必须从1级开始逐层展开，维度组合不能缺少上级
exp_dim_list=[['channel_level_1'],
              ['channel_level_1','channel_level_2'],
              ['channel_level_1','channel_level_2','channel_level_3'],
              ['channel_level_1','channel_level_2','channel_level_3','channel_level_4'],
              ['channel_level_1','channel_level_2','teacher_priority'],
              ['channel_level_1','channel_level_2','teacher_priority','teacher_location'],
              ['channel_level_1','channel_level_2','teacher_priority','teacher_location','father_depart_id'],
              ['channel_level_1','channel_level_2','teacher_priority','teacher_location','father_depart_id','depart_id'],
              ['channel_level_1','channel_level_2','teacher_location'],
              ['channel_level_1','channel_level_2','teacher_location','father_depart_id'],
              ['channel_level_1','channel_level_2','teacher_location','father_depart_id','depart_id'],
              ['channel_level_1','channel_level_2','channel_level_3'],
              ['channel_level_1','channel_level_2','channel_level_3','teacher_location'],
              ['channel_level_1','channel_level_2','channel_level_3','father_depart_id'],
              ['channel_level_1','channel_level_2','channel_level_3','father_depart_id','depart_id'],
              ['teacher_priority'],
              ['teacher_priority','teacher_location'],
              ['teacher_priority','teacher_location','father_depart_id'],
              ['teacher_priority','teacher_location','father_depart_id','depart_id'],
              ['teacher_priority','teacher_location','father_depart_id','channel_level_2'],
              ['teacher_priority','teacher_location','father_depart_id','channel_level_2','channel_level_3'],
              ['teacher_priority','channel_level_2'],
              ['teacher_priority','channel_level_2','teacher_location'],
              ['teacher_priority','channel_level_2','teacher_location','father_depart_id'],
              ['teacher_priority','channel_level_2','teacher_location','father_depart_id','depart_id'],
              ['teacher_priority','channel_level_2','channel_level_3'],
              ['teacher_priority','channel_level_2','channel_level_3','teacher_location'],
              ['teacher_priority','channel_level_2','channel_level_3','teacher_location','father_depart_id'],
              ['teacher_priority','channel_level_2','channel_level_3','teacher_location','father_depart_id','depart_id'],
              ['teacher_location'],
              ['teacher_location','father_depart_id'],
              ['teacher_location','father_depart_id','teacher_priority'],
              ['teacher_location','father_depart_id','depart_id'],
              ['teacher_location','father_depart_id','depart_id','teacher_priority'],
              ['teacher_location','father_depart_id','channel_level_2'],
              ['teacher_location','father_depart_id','channel_level_2','channel_level_3'],
              ['teacher_location','father_depart_id','depart_id','channel_level_2'],
              ['teacher_location','father_depart_id','depart_id','channel_level_2','channel_level_3']
              ]
train_dim_list=exp_dim_list[:]
train_dim_list.append(['buy_camp_money'])
for i in exp_dim_list:
    tmp_dim=['buy_camp_money']+i
    train_dim_list.append(tmp_dim)
#历史数据计算函数
##计算维度值
###筛选维度
def get_logic_dim_count(input_df,dim_list,camp_list):
    dim_dict={}
    c=0
    for d in camp_list:
        data_df=input_df[input_df['start_dayno']==d]
        dim_dict[d]={}
        for i in dim_list:
            tmp_dim_list=list(i)
            #tmp_dim_list.append('exp_start_dayno')
            c+=1
            #print(c,tmp_dim_list)
            if len(tmp_dim_list)==1:
                tmp_key=(tmp_dim_list[0])
            else:
                tmp_key=tuple(tmp_dim_list)
            dim_dict[d][tmp_key]={}
            #tmp_data_list.append(data_df[i])
            #分配用户数
            tmp_dim_dict=data_df.groupby(tmp_dim_list)['user_cnt'].sum().to_dict()
            dim_dict[d][tmp_key]['camp_user_cnt']=tmp_dim_dict
            #转化用户数
            tmp_dim_dict=data_df.groupby(tmp_dim_list)['is_transfer'].sum().to_dict()
            dim_dict[d][tmp_key]['transfer_user_cnt']=tmp_dim_dict
            #转化率
            tmp_dim_dict=(data_df.groupby(tmp_dim_list)['is_transfer'].sum()/data_df.groupby(tmp_dim_list)['user_cnt'].sum()).to_dict()
            dim_dict[d][tmp_key]['transfer_rate']=tmp_dim_dict
        #返回维度统计key的层次为：期次-维度名-指标名-维度值
        dim_dict[d][('exp_start_dayno')]={}
        dim_dict[d][('exp_start_dayno')]['camp_user_cnt']=data_df['user_cnt'].groupby(data_df['start_dayno']).sum().to_dict()
        dim_dict[d][('exp_start_dayno')]['transfer_user_cnt']=data_df['is_transfer'].groupby(data_df['start_dayno']).sum().to_dict()
        dim_dict[d][('exp_start_dayno')]['transfer_rate']=(data_df['is_transfer'].groupby(data_df['start_dayno']).sum()/data_df['user_cnt'].groupby(data_df['start_dayno']).sum()).to_dict()
    return dim_dict
##计算指标占比
def get_logic_dim_ratio(dim_dict,dim_list):
    for k in dim_dict.keys():
        for i in dim_list:
            if len(i)<2:
                up_dim=('exp_start_dayno')
                current_dict=dim_dict[k][(i[0])]
            elif len(i)==2:
                up_dim=(i[0])
                current_dict=dim_dict[k][tuple(i)]
            else:
                up_dim=tuple(i[:-1])
                current_dict=dim_dict[k][tuple(i)]
            up_dict=dim_dict[k][up_dim]
            current_dict['up_camp_user_cnt']={}
            current_dict['camp_user_ratio']={}
            current_dict['up_transfer_user_cnt']={}
            current_dict['transfer_user_ratio']={}
            camp_user_data=current_dict['camp_user_cnt']
            transfer_user_data=current_dict['transfer_user_cnt']
            for j in camp_user_data.keys():
                if len(i)==1:
                    up_key=k
                elif len(j)==2:
                    up_key=(j[0])
                else:
                    up_key=j[:-1]
                #print(up_key)
                current_dict['up_camp_user_cnt'][j]=up_dict['camp_user_cnt'][up_key]
                current_dict['camp_user_ratio'][j]=camp_user_data[j]/max(up_dict['camp_user_cnt'][up_key],1)
                current_dict['up_transfer_user_cnt'][j]=up_dict['transfer_user_cnt'][up_key]
                current_dict['transfer_user_ratio'][j]=transfer_user_data[j]/max(up_dict['transfer_user_cnt'][up_key],1)
            #camp_user_df=pd.DataFrame.from_dict(current_dict['camp_user_cnt'])
            if len(i)<2:
                dim_dict[k][(i[0])]=current_dict
            else:
                dim_dict[k][tuple(i)]=current_dict
    return dim_dict

dim_metrics_dict={}
for k in order_camp_dict_all.keys():
    #print(k)
    input_df=raw_df[(raw_df['school_subject']==k[0])&(raw_df['lesson_type']==k[1])]
    camp_list=set(input_df['start_dayno'].tolist())
    if k[1]=='exp':
        dim_list=exp_dim_list[:]
    if k[1]=='train':
        dim_list=train_dim_list[:]
    dim_dict=get_logic_dim_count(input_df,dim_list,camp_list)
    dim_metrics_dict[k]=get_logic_dim_ratio(dim_dict,dim_list)
print(dim_metrics_dict[('abc','exp')].keys())

from collections import Counter
import copy

#转化维度key,传入的是list
def dim_key_transfer(current_dim):
    if len(current_dim)<2:
        res_key=current_dim[0]
    else:
        res_key=tuple(current_dim)
    return res_key
#获取上级维度key，传入的是tuple
def get_up_dim_key(current_dim):
    if len(current_dim)==2:
        res_key=current_dim[0] 
    elif np.size(np.array((current_dim)))>2:
        res_key=current_dim[:-1]
    else:
        res_key='exp_start_dayno'
    return res_key
#获取历史数据日期
def get_his_date_key(start_week,cal_weeks,method='start'):
    if start_week not in camp_order_dict.keys():
        print('开营日期输入错误，请检查您的输入（2周/3周体验课的开营日期是周一；1周体验课的开营日期是周日）')
        return 'error'
    start_order=camp_order_dict.get(start_week)
    res_date=[]
    if method=='start':
        for i in range(int(start_order),int(start_order+cal_weeks)):
            res_date.append(order_camp_dict.get(i))
        return res_date
    else:
        for i in range(int(start_order-cal_weeks),int(start_order)):
            res_date.append(order_camp_dict.get(i))
        return res_date

#历史数据汇总
def his_data_sum(dim_dict,dim_list,start_week,cal_weeks,method='start'):
    dim_dict=copy.deepcopy(dim_dict)
    his_date=get_his_date_key(start_week,cal_weeks,method)
    base_dict=dim_dict[his_date[0]].copy()
    #print(his_date[0])
    if len(his_date)==1:
        return base_dict
    for d in his_date[1:]:
        #print(d)
        add_dict=dim_dict[d].copy()
        for i in base_dict.keys():
            for j in base_dict[i].keys():
                x=Counter(base_dict[i][j].copy())
                y=Counter(add_dict[i][j].copy())
                base_dict[i][j]=dict(x+y).copy()
    return base_dict
#计算JS
def cal_js(p,q):
    if q==0:
        q=0.000001
    if p==0:
        p=0.000001
    res=50*(p*np.log10(2*p/(p+q))+q*np.log10(2*q/(p+q)))
    return res
#计算贡献度,原始公式
def cal_attribution(a1,f1,A1,F1,a2,f2,A2,F2):
    res=(F2*(a1-f1)-F1*(a2-f2))/(F2*(F2+a2-f2))
    return res
#计算贡献度，变化公式
#计算变动率
def cal_change_rate(a,b):
    res=a/(b+1e-6)-1
    return res

#按维度计算转化率
#输入：dim_dict-当期数据；his_dim_dict-历史数据；dim-计算维度；dim_values-维度值
#输出：forcast_transfer_rate-历史转化率；exp_transfer_rate-预期转化率；actual_transfer_rate-实际转化率
def get_dim_transfer_rate(dim_dict,his_dim_dict,dim,dim_values='all'):
    forcast_transfer_rate=0
    exp_transfer_rate=0
    actual_transfer_rate=0
    up_dim=get_up_dim_key(dim)
    for k in dim_dict[dim]['camp_user_cnt'].keys():
        up_k=get_up_dim_key(k)
        if dim_values!='all' and up_k!=dim_values:
            continue
        #当期分配人数占比
        c0=dim_dict[dim]['camp_user_ratio'][k]
        #历史分配人数占比
        c1=his_dim_dict[dim]['camp_user_cnt'].get(k,0)/his_dim_dict[dim]['up_camp_user_cnt'].get(k,1) 
        #当期转化率
        t0=dim_dict[dim]['transfer_rate'][k] 
        #历史转化率，如果当前维度没有，要取上一个维度的数据
        t1a=his_dim_dict[dim]['transfer_user_cnt'].get(k,0)
        t1b=his_data[dim]['camp_user_cnt'].get(k,1)
        if t1b<30:
            t1a=his_dim_dict[up_dim]['transfer_user_cnt'].get(up_k,0)
            t1b=his_data[up_dim]['camp_user_cnt'].get(up_k,1)
        t1=t1a/t1b
        actual_transfer_rate+=c0*t0
        exp_transfer_rate+=c0*t1
        forcast_transfer_rate+=c1*t1
    if forcast_transfer_rate==0:
        forcast_transfer_rate=t1
    return forcast_transfer_rate,exp_transfer_rate,actual_transfer_rate

#按维度计算JS
#输入：dim_dict-当期数据；his_dim_dict-历史数据；dim-计算维度；dim_values-维度值
#输出：js_u-分配人数分布变化；js_t-转化人数分布变化；js_all总惊喜度
def get_dim_js(dim_dict,his_dim_dict,dim,dim_values='all'):
    #if dim_values=='all':
    #    dim_values=dim_dict[dim]['camp_user_cnt'].keys()
    js_u=0
    js_t=0
    for k in dim_dict[dim]['camp_user_cnt'].keys():
        if dim_values!='all' and get_up_dim_key(k)!=dim_values:
            continue
        #计算分配人数占比散度
        p=dim_dict[dim]['camp_user_ratio'][k] #当期分配人数占比
        q=his_dim_dict[dim]['camp_user_cnt'].get(k,0)/(his_dim_dict[dim]['up_camp_user_cnt'].get(k,1)+1e-6) #历史分配人数占比
        js_u+=cal_js(p,q)
        #计算支付占比JS变化
        p=dim_dict[dim]['transfer_user_ratio'][k]
        q=his_dim_dict[dim]['transfer_user_cnt'].get(k,0)/(his_dim_dict[dim]['up_transfer_user_cnt'].get(k,1)+1e-6) 
        js_t+=cal_js(p,q)
    js_all=js_t+js_u
    return js_u,js_t,js_all

#按维度计算贡献度
#输入：dim_dict-当前期次数据；his_dim_dict-历史期次数据；dim-计算维度；forecast-历史转化率；exp-期望转化率；actual-实际转化率；dim_values-要计算的维度值，默认为all,即全部维度
#输出：ratio_dict-流量占比变化贡献度；rate_dict-转化率变化贡献度；ratio_inf_dict-流量占比变化具体值；rate_inf_dict-转化率变化具体值
def get_dim_att(dim_dict,his_dim_dict,dim,forecast,exp,actual,dim_values='all'):
    #记录贡献度的值
    ratio_dict={}
    rate_dict={}
    #记录对转化率影响的值，用于解释结果
    ratio_inf_dict={}
    rate_inf_dict={}
    up_dim=get_up_dim_key(dim)
    for k in dim_dict[dim]['camp_user_cnt'].keys():
        up_k=get_up_dim_key(k)
        if dim_values!='all' and up_k!=dim_values:
            continue
        #计算分配人数占比散度
        #当期分配人数占比
        p=dim_dict[dim]['camp_user_ratio'][k] 
        #历史分配人数占比
        q=his_dim_dict[dim]['camp_user_cnt'].get(k,0)/his_dim_dict[dim]['up_camp_user_cnt'].get(k,1) 
        #当期转化率
        a=dim_dict[dim]['transfer_rate'][k] 
        #历史转化率，如果当前维度没有，要取上一个维度的数据
        ft=his_dim_dict[dim]['transfer_user_cnt'].get(k,0)
        fc=his_data[dim]['camp_user_cnt'].get(k,1)
        if fc<30:
            ft=his_dim_dict[up_dim]['transfer_user_cnt'].get(up_k,0)
            fc=his_data[up_dim]['camp_user_cnt'].get(up_k,1)
        f=ft/fc#历史转化率
        ratio_dict[k]=int(f*(p-q)/(exp-forecast+1e-6)*1e4)/1e4
        ratio_inf_dict[k]=(int(p*1e4)/100,int(q*1e4)/100,int(f*(p-q)*1e4)/1e2,int(f*(p-q)/(exp-forecast+1e-6)*1e4)/1e2)
        rate_dict[k]=int((a-f)*p/(actual-exp+1e-6)*1e4)/1e4
        rate_inf_dict[k]=(int(a*1e4)/100,int(f*1e4)/100,int((a-f)*p*1e4)/1e2,int((a-f)*p/(actual-exp+1e-6)*1e4)/1e2)
    return ratio_dict,rate_dict,ratio_inf_dict,rate_inf_dict

def get_dict_value(x,d):
    if np.size(x)==1:
        res_list=d.get(x,x)
        return res_list,res_list
    res_list=[]
    res_content=''
    for i in x:
        res_list.append(d.get(i,i))
        res_content=res_content+d.get(i,i)+'-'
    return res_list,res_content[:-1]
def get_dim_level(x,l):
    if np.size(x)==1:
        if l==1:
            return x
        return -1
    if l<=np.size(x):
        return x[l-1] 
    return -1

def reason_search(current_data,his_data,search_tree,t_dict,dim_dict,dim_level,result_dim,cc_direction,top_reason=1):
    res_dim={}
    dim_v='all'
    dim_v_all=[dim_v]
    search_dim=search_tree[dim_level]
    for d in search_dim:
        res_key=['first_level']
        if dim_level>1:
            up_level=dim_level-1
            up_dim=get_up_dim_key(d) 
            res_key=[]
            if up_dim not in result_dim[up_level].keys():
                continue
            dim_v_all=set()
            for tf in ('rate_dim','ratio_dim'):
                if result_dim[up_level][up_dim].get(tf,{})!={}:
                    dim_v_all=dim_v_all.union(result_dim[up_level][up_dim][tf].keys())
        if(len(dim_v_all)==0):
            continue
        #print('当前搜索维度：',d)
        for dim_v in dim_v_all:
            #print('当前维度值',dim_dict[0].get(dim_v,dim_v))
            #计算预期转化率
            f_rate,e_rate,a_rate=get_dim_transfer_rate(dim_dict=current_data,his_dim_dict=his_data,dim=d,dim_values=dim_v)
            #print('当前期次维度转化率',int(a_rate*1e4)/1e2,'%')
            #print('维度预期转化率',int(e_rate*1e4)/1e2,'%')
            #print('历史期次维度转化率',int(f_rate*1e4)/1e2,'%')
            change_rate=cal_change_rate(a_rate,f_rate)
            change_direction=print_direction(change_rate)
            #计算JS散度
            js_u,js_t,js_all=get_dim_js(dim_dict=current_data,his_dim_dict=his_data,dim=d,dim_values=dim_v)
            if js_all>t_dict['js_threshold']: #JS超过阈值，再继续计算
                if d not in res_dim.keys():
                    res_dim[d]={}
                #print(print_over_threshold(d,js_all,t_dict['js_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],metrics='惊喜度',comment='，将继续在此维度下归因'))
                #print(print_dim_metrics(d,metrics='预期转化率',value=e_rate,name_dict=dim_dict[0]))
                percentage,rate,percentage_inf,rate_inf=get_dim_att(current_data,his_data,d,f_rate,e_rate,a_rate,dim_values=dim_v)
                percentage_dim_values={}
                rate_dim_values={}
                percentage_filter_dict={}
                rate_filter_dict={}
                for k in percentage.keys():
                    #print(k,percentage_inf[k],rate_inf[k])
                    d_direction=print_direction(percentage_inf[k][2])
                    if abs(percentage[k])>t_dict['ratio_total_threshold'] and percentage[k]>0 and d_direction==cc_direction:
                        percentage_dim_values[k]=[percentage[k],percentage_inf[k],rate_inf[k],[int(f_rate*1e4)/1e2,int(e_rate*1e4)/1e2,int(a_rate*1e4)/1e2]]
                        #print(print_over_threshold(d,percentage[k],t_dict['ratio_total_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],k,metrics='占比变化贡献度'))
                        #print(print_over_threshold(d,percentage_inf[k],t_dict['ratio_total_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],k,metrics='流量占比',method='change',c_direction=print_direction(percentage_inf[k][0]-percentage_inf[k][1])))
                        #print(print_over_threshold(d,percentage_inf[k],t_dict['ratio_total_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],k,metrics='转化率',method='ratio',c_direction=print_direction(percentage_inf[k][0]-percentage_inf[k][1])))
                    elif abs(percentage[k])>t_dict['ratio_single_threshold'] and percentage[k]>0 and d_direction==cc_direction:
                        percentage_filter_dict[k]=percentage[k]
                    d_direction=print_direction(rate_inf[k][2])
                    if abs(rate[k])>t_dict['rate_total_threshold'] and rate[k]>0 and d_direction==cc_direction:
                        rate_dim_values[k]=[rate[k],percentage_inf[k],rate_inf[k],[int(f_rate*1e4)/1e2,int(e_rate*1e4)/1e2,int(a_rate*1e4)/1e2]]
                        #print(print_over_threshold(d,rate[k],t_dict['rate_total_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],k,metrics='转化率变化贡献度'))
                        #print(print_over_threshold(d,rate_inf[k],t_dict['rate_total_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],k,metrics='转化率',method='change',c_direction=print_direction(rate_inf[k][0]-rate_inf[k][1])))
                        #print(print_over_threshold(d,rate_inf[k],t_dict['rate_total_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],k,metrics='转化率',method='rate',c_direction=print_direction(rate_inf[k][0]-rate_inf[k][1])))
                    elif abs(rate[k])>t_dict['rate_single_threshold'] and rate[k]>0 and d_direction==cc_direction:
                        rate_filter_dict[k]=rate[k]
                if len(percentage_dim_values)<top_reason:
                    if sum(percentage_filter_dict.values())>t_dict['ratio_total_threshold'] or np.size(percentage_filter_dict)==1:
                        tmp_list=sorted(percentage_filter_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
                        for u in enumerate(tmp_list):
                            if u[0]>2:
                                break
                            m=u[1][0]
                            percentage_dim_values[m]=[percentage[m],percentage_inf[m],rate_inf[m],[int(f_rate*1e4)/1e2,int(e_rate*1e4)/1e2,int(a_rate*1e4)/1e2]]
                            #print(print_over_threshold(d,percentage_inf[m],t_dict['ratio_single_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],m,metrics='流量占比',method='change',c_direction=print_direction(percentage_inf[m][0]-percentage_inf[m][1])))
                            #print(print_over_threshold(d,percentage_inf[m],t_dict['ratio_single_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],m,metrics='转化率',method='ratio',c_direction=print_direction(percentage_inf[m][0]-percentage_inf[m][1])))
                    
                if len(rate_dim_values)<top_reason:
                    if sum(rate_filter_dict.values())>t_dict['rate_total_threshold'] or np.size(rate_filter_dict)==1:
                        tmp_list=sorted(rate_filter_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
                        for u in enumerate(tmp_list):
                            if u[0]>2:
                                break
                            m=u[1][0]
                            rate_dim_values[m]=[rate[m],percentage_inf[m],rate_inf[m],[int(f_rate*1e4)/1e2,int(e_rate*1e4)/1e2,int(a_rate*1e4)/1e2]]
                            #print(print_over_threshold(d,rate_inf[m],t_dict['rate_single_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],m,metrics='转化率',method='change',c_direction=print_direction(rate_inf[m][0]-rate_inf[m][1])))
                            #print(print_over_threshold(d,rate_inf[m],t_dict['rate_single_threshold'],dim_dict[0],dim_dict[1],dim_dict[2],m,metrics='转化率',method='rate',c_direction=print_direction(rate_inf[m][0]-rate_inf[m][1])))
                    else:
                        1
                        #print('当前维度下没有转化率发生明显变化的维度项')
                if percentage_dim_values!={}:
                    if 'ratio_dim' not in res_dim[d].keys():
                        res_dim[d]['ratio_dim']={}
                    res_dim[d]['ratio_dim'].update(copy.deepcopy(percentage_dim_values))
                else:
                    1
                    #print('当前维度下没有流量占比发生明显变化的维度项')
                if rate_dim_values!={}:         
                    if 'rate_dim' not in res_dim[d].keys():
                        res_dim[d]['rate_dim']={}
                    res_dim[d]['rate_dim'].update(copy.deepcopy(rate_dim_values)) 
                else:
                    1
                    #print('当前维度下没有转化率发生明显变化的维度项')
                #if 'transfer_ratio' not in res_dim[d].keys():
                #    res_dim[d]['transfer_ratio']={}
                #res_dim[d]['transfer_ratio'][dim_v]=[int(f_rate*1e4)/1e2,int(e_rate*1e4)/1e2,int(a_rate*1e4)/1e2]
            else:
                1
                #print('当前维度惊喜度为'+str(int(js_all*1e4)/1e4)+'，未超出惊喜度阈值，停止搜索')
        #print('\n')
    result_dim[dim_level]=res_dim
    return result_dim
def get_result_df(result_reason_dim):
    result_reason_df=pd.DataFrame(columns=['level','dim','reason_type','dim_value','metrics','ratio_change','rate_change','dim_rate'])
    for l in result_reason_dim.items():
        for c in l[1].items():
            for m in c[1].items():
                for v in m[1].items():
                    #print(l[0],c[0],m[0],v[0],v[1][0],v[1][1],v[1][2],v[1][3])
                    result_reason_df.loc[-1]=[l[0],c[0],m[0],v[0],v[1][0],v[1][1],v[1][2],v[1][3]]
                    result_reason_df.index = result_reason_df.index+1
                    result_reason_df = result_reason_df.sort_index() 
    result_reason_df['dim_name']=result_reason_df['dim'].apply(lambda x: get_dict_value(x,dimension_name_dict)[1])
    result_reason_df['value_name']=result_reason_df['dim_value'].apply(lambda x: get_dict_value(x,dimension_name_dict)[1])
    result_reason_df['dim_1']=result_reason_df['dim'].apply(lambda x: get_dict_value(get_dim_level(x,1),dimension_name_dict)[1])
    result_reason_df['dim_2']=result_reason_df['dim'].apply(lambda x: get_dict_value(get_dim_level(x,2),dimension_name_dict)[1])
    result_reason_df['dim_3']=result_reason_df['dim'].apply(lambda x: get_dict_value(get_dim_level(x,3),dimension_name_dict)[1])
    result_reason_df['dim_4']=result_reason_df['dim'].apply(lambda x: get_dict_value(get_dim_level(x,4),dimension_name_dict)[1])
    result_reason_df['dim_5']=result_reason_df['dim'].apply(lambda x: get_dict_value(get_dim_level(x,5),dimension_name_dict)[1])
    result_reason_df['dim_6']=result_reason_df['dim'].apply(lambda x: get_dict_value(get_dim_level(x,6),dimension_name_dict)[1])
    for i in range(1,7):
        col_name='value_'+str(i)
        dim_name='dim_'+str(i)
        if '老师大组' in result_reason_df[dim_name].unique() or '老师小组' in result_reason_df[dim_name].unique():
            tmp_dict=check_dict
            result_reason_df['value_name']=result_reason_df['dim_value'].apply(lambda x: get_dict_value(x,tmp_dict)[1])
        else:
            tmp_dict=dimension_name_dict
        result_reason_df[col_name]=result_reason_df['dim_value'].apply(lambda x: get_dict_value(get_dim_level(x,i),tmp_dict)[1])
    return result_reason_df
def get_child_key(key_dict,x,u,res=[]):
    for k in key_dict.keys():
        if np.size(k)==x:
            if np.size(k)==1 and k not in res:
                res.append(k)
                s=[k]
            elif get_up_dim_key(k)==u:
                s=[u]
                s.append(k)
                res.append(s)
            res=get_child_key(key_dict,x=x+1,u=k,res=res)
    return res
#特殊样式打印
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
#x：打印纸
#b：是否加粗
#u：是否加下划线
#c：打印颜色，使用color下的属性RED/YELLOW/GREEN/BLUE
def print_format(x,b=False,u=False,c=''):
    if b==True:
        x=color.BOLD+x
    if u==True:
        x=color.UNDERLINE+x
    if c!='':
        x=c+x
    return x+color.END

#变化方向打印
def print_direction(x,acc=1e-6):
    if x>acc:
        return '上升'
    if x<-acc:
        return '下降'
    return '持平'

#整体转化率变化打印
#c_camp：当前期次
#c_rate：当前转化率
#h_rate：历史转化率
#h_camp：历史期次
#camps：历史比较期数
#method：统计方法 start-从历史期次向后统计；end-从当前期次向前统计
def print_camp_rate(c_camp,c_rate,h_rate=0,h_camp=20200706,camps=1,method='start'):
    change_rate=cal_change_rate(c_rate,h_rate)
    change_direction=print_direction(change_rate)
    print_c=color.BLUE
    c=print_format(str(c_camp),b=True,c=print_c)+'期转化率为'+print_format(str(int(c_rate*1e4)/1e2),b=True,c=print_c)+print_c+'%，'+color.END
    if method=='start':
        m='相比于从'+str(h_camp)+'开始的'+str(camps)+'期的平均转化率'+str(int(h_rate*10000)/100)+'%，'
    else:
        m='相比于过往'+str(camps)+'期的平均转化率'+str(int(h_rate*10000)/100)+'%，'
    if change_direction=='下降':
        print_c=color.RED
    if change_direction=='上升':
        print_c=color.GREEN
    res=c+m+print_format(change_direction,c=print_c)+print_format(str(abs(int(change_rate*1e4)/1e2)),b=True,c=print_c)+print_c+'%'+color.END
    return res

#超出维度阈值的打印
#d：当前维度
#m：当前指标值
#t：当前阈值
#dim_name_dict：维度名称字典
#dim_value_dict：维度名称简称字典
#check_dict：维度值字典
#metrics：指标名称
#dim_value：维度值
#comment：附加说明
#method:输出比较方式 threshold-阈值；change-指标变化；all-整体影响
#c_direction:指标变化方向
def print_over_threshold(d,m,t,dim_name_dict,dim_value_dict={},check_dict={},dim_value='',metrics='惊喜度',comment='',method='threshold',c_direction=''):
    up_name=''
    up_d=get_up_dim_key(d)   
    if np.size(d)>1:
        tmp_d=d[-1]
        if np.size(up_d)==1:
            up_name=dim_name_dict.get(up_d,up_d)+'-'
        else:
            for x in np.array(up_d):
                up_name+=dim_name_dict.get(x,x)
                up_name+='-'
    else:
        tmp_d=d
    res=up_name+dim_name_dict.get(tmp_d,tmp_d)+'维度'
    if dim_value!='':
        if np.size(d)>1:
            v=''
            for u in dim_value:
                tmp_v=u
                v=v+check_dict.get(tmp_v,dim_name_dict.get(tmp_v,tmp_v))+'-'
            v=v[:-1]
        else:
            tmp_v=dim_value
            v=check_dict.get(tmp_v,dim_name_dict.get(tmp_v,tmp_v))
        n=dim_value_dict.get(tmp_d,'')
        res=res+'下，'+print_format(v,b=True,c=color.RED)+n+'的'
    res=res+metrics
    if method=='threshold':
        res=res+'为'+str(int(m*1e4)/1e4)+'，超出'+metrics+'阈值'+str(t)+comment
    elif method=='change':
        res=res+'从'+str(m[1])+'%'+c_direction+'到'+str(m[0])+'%'
    elif method=='ratio':
        res='造成'+up_name+'预期'+metrics+c_direction+str(abs(m[2]))+'%，占'+metrics+'变化的'+str(m[3])+'%'
    elif method=='rate':
        res='造成'+up_name+'实际'+metrics+c_direction+str(abs(m[2]))+'%，占'+metrics+'变化的'+str(m[3])+'%'
    elif method=='all':
        res='造成'+up_name+'整体'+metrics+c_direction+str(abs(m[2]))+'%，占'+metrics+'变化的'+str(m[3])+'%'
    return res

#打印维度指标值
def print_dim_metrics(d,metrics,value,name_dict):
    u_name=''
    up_d=get_up_dim_key(d)
    if np.size(d)>1:
        tmp_d=d[-1]
        for x in np.array(up_d):
            u_name+=name_dict.get(x,x)
            u_name+='-'
    else:
        tmp_d=d
    res=u_name+name_dict.get(tmp_d,tmp_d)+metrics+'为'+str(int(value*1e4)/1e2)+'%'
    return res

dimension_name_dict={'channel_level_1':'一级渠道','channel_level_2':'二级渠道','channel_level_3':'三级渠道',
                     'channel_level_4':'四级渠道','sku':'sku','chan_name':'班级类别','teacher_priority':'老师圈层',
                     'father_depart_id':'老师大组','depart_id':'老师小组',
                     'adv':'广告','wx':'微信大投放','wx-old':'微信广告','gdt':'广点通','dy':'抖音','bd':'百度',
                     'out':'合作渠道','teacher':'老师转化','kol':'KOL','outbj':'北京渠道','company':'公司合作','activity':'活动',
                     'zrllgzh':'公众号','app':'内容锁','other':'其它','fx':'分销','dx':'电销','xcx':'小程序','sharexcx':'小程序分享',
                     'gzhpush':'公众号推送','qyy':'群运营','kuoke':'扩科','nature':'自然流量','reco':'转介绍',
                     'ad':'品牌广告','campaign':'品牌广告','baidu':'百度','in':'内部合作','empty':'空渠道','unknown':'未知渠道',
                     'all':'全部','teacher_location':'老师所属基地'
                    }
dimension_value_dict={'channel_level_1':'渠道','channel_level_2':'渠道','channel_level_3':'渠道',
                     'channel_level_4':'渠道','sku':'商品','chan_name':'班级','teacher_priority':'圈层',
                      'teacher_location':'老师所属基地','buy_camp_money':'入营金额'}

search_tree_all={}
for t in ('exp','train'):
    if t=='exp':
        cal_dim_list=exp_dim_list
    if t=='train':
        cal_dim_list=train_dim_list
    search_tree={}
    for i in cal_dim_list:
        if i[-1]=='sku' or 'chan_name' in i:
            continue
        x= np.size(np.array((i)))
        if x not in search_tree.keys():
                search_tree[x]=[]
        search_tree[x].append(dim_key_transfer(i))
    search_tree_all[t]=search_tree

#输入参数
#school='abc'
#lesson_level='s1'
teacher_group_dict={}
his_method='end'
js_threshold=0.4295 #js阈值
ratio_single_threshold=1.3 #结构变化，单一维度值阈值
ratio_total_threshold=2.68 #结构变化，多维度值阈值
rate_single_threshold=0.3 #非结构变化，单一维度值阈值
rate_total_threshold=0.6 #非结构变化，多维度值阈值
search_depth=1 #搜索深度
#数据准备
threshold_dict={'js_threshold':js_threshold,'ratio_single_threshold':ratio_single_threshold,
                'ratio_total_threshold':ratio_total_threshold,'rate_single_threshold':rate_single_threshold,
                'rate_total_threshold':rate_total_threshold}
check_dict=copy.deepcopy(teacher_group_dict)
dim_dict_combine=[dimension_name_dict,dimension_value_dict,check_dict]

all_result_reason_df=pd.DataFrame()
#按科目与课程类型计算
for k in dim_metrics_dict.keys():
    school_subject=k[0]
    lesson_type=k[1]
    search_tree=search_tree_all[lesson_type]
    history_dict=dim_metrics_dict[k]
    order_camp_dict=order_camp_dict_all[k]
    camp_order_dict=camp_order_dict_all[k]
    current_start_dayno=max(order_camp_dict.values())
    #current_start_dayno='20210322'
    his_start_dayno='20210301'
    current_data=history_dict[current_start_dayno]
    current_transfer_rate=current_data['exp_start_dayno']['transfer_rate'][current_start_dayno]
    print(k,current_start_dayno)
    #计算与不同往期期数的比较
    #for his_camps in range(1,3):
    for his_camps in range(1,min(len(dim_metrics_dict[k].keys()),9)):
        if his_method=='start':
            his_data=his_data_sum(history_dict,cal_dim_list,his_start_dayno,his_camps,his_method)
        else:
            his_data=his_data_sum(history_dict,cal_dim_list,current_start_dayno,his_camps,his_method)
        if his_data!='error':
            #print(his_data['exp_start_dayno'])
            his_t=sum(his_data['exp_start_dayno']['transfer_user_cnt'].values())
            his_u=sum(his_data['exp_start_dayno']['camp_user_cnt'].values())
            print(his_t)
            print(his_u)
            his_transfer_rate=float(his_t)/float(his_u)
            print(current_transfer_rate,his_transfer_rate)
            transfer_rate_change=cal_change_rate(current_transfer_rate,his_transfer_rate)
            a_rate=current_transfer_rate
            f_rate=his_transfer_rate
            dim_dict_combine=[dimension_name_dict,dimension_value_dict,check_dict]
            #归因搜索
            #print('当前期次转化率',int(current_transfer_rate*1e4)/1e2,'%，历史期次转化率',int(his_transfer_rate*1e4)/1e2,'%，转化率变动率',int(transfer_rate_change*1e4)/1e2,'%')
            camp_change_direction=print_direction(transfer_rate_change)
            print(print_camp_rate(current_start_dayno,current_transfer_rate,his_transfer_rate,his_start_dayno,his_camps,his_method))
            if lesson_type=='train':
                search_depth=6
            else:
                search_depth=5
            middle_dim={}
            result_reason_dim={}
            for le in range(1,search_depth+1):
                if le in search_tree.keys():
                    middle_dim=reason_search(current_data,his_data,search_tree,threshold_dict,dim_dict_combine,le,middle_dim,camp_change_direction,3) 
                    if middle_dim[le]=={}:     
                        break
                    result_reason_dim[le]=middle_dim[le]
            result_reason_df=get_result_df(result_reason_dim)
            result_reason_df['school_subject']=school_subject
            result_reason_df['lesson_type']=lesson_type
            result_reason_df['start_dayno']=current_start_dayno
            result_reason_df['his_camps']=his_camps
            print(result_reason_df.shape[0])
            all_result_reason_df=pd.concat([all_result_reason_df,result_reason_df])

#school_subject 科目
#lesson_type 课程类型
#current_start_dayno 当前计算日期
#his_camps 历史期次比较数量

#result_reason_dim结构:
#key 归因层级-归因维度-归因类型（结构、非结构）-维度值
#value 贡献度,比例变化影响(当期分配人数占比，历史分配人数占比，对预期转化率的影响值，对预期转化率变化的贡献度),
#转化率变化影响(当期转化率，历史转化率，对实际转化率的影响值，对实际转化率变化的贡献度),维度转化率变化[历史转化率，预期转化率，实际转化率]

final_result_df=pd.DataFrame()
final_result_df['ftime']=YYYYMMDD
final_result_df[['school_subject','lesson_type','start_dayno','his_camps','search_level']]=all_result_reason_df[['school_subject','lesson_type','start_dayno','his_camps','level']]
final_result_df[['reason_type','metrics','dim_1','dim_2','dim_3','dim_4','dim_5','dim_6']]=all_result_reason_df[['reason_type','metrics','dim_1','dim_2','dim_3','dim_4','dim_5','dim_6']]
final_result_df[['value_1','value_2','value_3','value_4','value_5','value_6']]=all_result_reason_df[['value_1','value_2','value_3','value_4','value_5','value_6']]
final_result_df['up_a_rate']=all_result_reason_df['dim_rate'].apply(lambda x:x[2])
final_result_df['up_e_rate']=all_result_reason_df['dim_rate'].apply(lambda x:x[1])
final_result_df['up_f_rate']=all_result_reason_df['dim_rate'].apply(lambda x:x[0])
final_result_df['a_percentage']=all_result_reason_df['ratio_change'].apply(lambda x:x[0])
final_result_df['f_percentage']=all_result_reason_df['ratio_change'].apply(lambda x:x[1])
final_result_df['percentage_change']=all_result_reason_df['ratio_change'].apply(lambda x:x[2])
final_result_df['percentage_change_ratio']=all_result_reason_df['ratio_change'].apply(lambda x:x[3])
final_result_df['a_rate']=all_result_reason_df['rate_change'].apply(lambda x:x[0])
final_result_df['f_rate']=all_result_reason_df['rate_change'].apply(lambda x:x[1])
final_result_df['rate_change']=all_result_reason_df['rate_change'].apply(lambda x:x[2])
final_result_df['rate_change_ratio']=all_result_reason_df['rate_change'].apply(lambda x:x[3])
final_result_df['ftime']=YYYYMMDD
final_result_df.reset_index(inplace=True)
final_result_df.drop(columns='index',inplace=True)
#final_result_df.to_csv('final_result_df.csv')

for c in final_result_df.columns:
    final_result_df[c]=final_result_df[c].astype(str)
t_data=[]
for i in final_result_df.index:
    t_data.append(tuple(final_result_df.loc[i,:]))
t_schema = ('ftime', 'school_subject', 'lesson_type', 'start_dayno', 'his_camps',\
       'search_level', 'reason_type', 'metrics', 'dim_1', 'dim_2', 'dim_3',\
       'dim_4', 'dim_5', 'dim_6', 'value_1', 'value_2', 'value_3', 'value_4',\
       'value_5', 'value_6', 'up_a_rate', 'up_e_rate', 'up_f_rate',\
       'a_percentage', 'f_percentage', 'percentage_change',\
       'percentage_change_ratio', 'a_rate', 'f_rate', 'rate_change',\
       'rate_change_ratio')

output_df=spark.createDataFrame(t_data,t_schema)
tb_name='abc_dws_sale_transfer_rate_reasoning_inc_w'
provider.saveToTable(output_df,tb_name,'p_'+YYYYMMDD)