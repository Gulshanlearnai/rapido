# This file primarily contains two modules:

# olf_postprocess
# Handles order-level feature engineering, including: LM binning, Peak hour tagging, TTC / TTA metrics, Bid tagging, Order status flags, PPKM calculation

# olf_metrics 
# Performs multi-level aggregations with: G2N funnel metrics, TTA / TTC percentiles, Bid order % and bid amount percentiles, Ping metrics (APR, DPR, ME, etc.)

import numpy as np
import pandas as pd


def lm_bin(x: float) -> str:
    """
    Categorize trip distance into bins.

    Parameters
    ----------
    x : float
        Trip distance in km.

    Returns
    -------
    str
        short / medium / long / Other
    """
    if 0 <= x < 5:
        return "short"
    elif 5 <= x < 10:
        return "medium"
    elif x >= 10:
        return "long"
    return "Other"


def tag_peak_period(qh: str) -> str:
    """Tag basic peak periods."""
    if "0800" <= qh <= "1100":
        return "morning_peak"
    elif "1700" <= qh <= "2000":
        return "evening_peak"
    return "rest"


def tag_peak_period_v1(qh: str) -> str:
    """More granular peak tagging."""
    if "0000" <= qh < "0800":
        return "rest_morning"
    elif "0800" <= qh < "1100":
        return "morning_peak"
    elif "1100" <= qh < "1700":
        return "afternoon"
    elif "1700" <= qh < "2000":
        return "evening_peak"
    return "rest_evening"


def olf_postprocess(df):

    # df['service_level'] = df['service_detail_id'].map(service_mapping)
    df['quarter_hour'] = df['quarter_hour'].astype(str).str.strip()  # remove spaces
    
    df['peak_hour_tag'] = df['quarter_hour'].apply(tag_peak_period)
    df['hour_tag'] = df['quarter_hour'].apply(tag_period)
    df['lm_bin'] = df['distance_final_distance'].apply(lm_bin)
    
    
    df['cobra_ttc'] = np.where(df['modified_order_status'] == 'COBRA',
                                     (df['customer_cancelled_epoch'] - df['order_requested_epoch']) / 1000, 
                                     np.nan)
    
    # OCARA TTC (cancelled after acceptance)
    df['ocara_ttc'] = np.where(df['modified_order_status'] == 'OCARA',
                                    (df['customer_cancelled_epoch'] - df['accepted_epoch']) / 1000,
                                     np.nan)
    
    # TTA (accepted after request)
    df['tta'] = np.where(df['accepted_epoch'].notna(),
                              (df['accepted_epoch'] - df['order_requested_epoch']) / 1000,
                               np.nan)
    
    # Precompute ppkm
    df['ppkm'] = np.where(df['distance_final_distance'] > 0,
                                df['amount'] / df['distance_final_distance'],
                                np.nan)

    df["is_accepted"] = np.where(df["accepted_epoch"].notna(), df['order_id'], np.nan)
    df["is_dropped"]  = np.where(df["modified_order_status"] == "dropped", df['order_id'], np.nan)
    df["is_cobrm"]    = np.where(df["modified_order_status"] == "COBRM", df['order_id'], np.nan)
    df["is_cobra"]    = np.where(df["modified_order_status"] == "COBRA", df['order_id'], np.nan)
    df["is_ocara"]    = np.where(df["modified_order_status"] == "OCARA", df['order_id'], np.nan)
    df["is_expired"]  = np.where(df["modified_order_status"] == "expired", df['order_id'], np.nan)

    df["is_positive_bid"]  = np.where(df["bid_type"] == "positive bid", df['order_id'], np.nan)
    df["is_negative_bid"]  = np.where(df["bid_type"] == "negative bid", df['order_id'], np.nan)
    df["is_zero_bid"]  = np.where(df["bid_type"] == "zero bid", df['order_id'], np.nan)

    df['pos_amount_bid']  = np.where(df['bid_type'] == 'positive bid', df['amount_breakup_bid_delta_total'], np.nan)
    df['neg_amount_bid']  = np.where(df['bid_type'] == 'negative bid', df['amount_breakup_bid_delta_total'], np.nan)
    df['zero_amount_bid'] = np.where(df['bid_type'] == 'zero bid', df['amount_breakup_bid_delta_total'], np.nan)

    return df


def olf_metrics(df, group_cols, dropna=False):
    """
    Replicates SQL aggregations from the orders table using pandas.
    Parameters:
        df: DataFrame to aggregate
        group_cols: list of columns to group by
    Returns:
        aggregated DataFrame
    """
    def safe_percentile(x, q):
        x = x.dropna()
        return np.round(np.percentile(x, q),2) if len(x) > 0 else np.nan

    grp = df.groupby(group_cols, dropna= dropna)

    agg_df = grp.agg(
        gross_customers=('customer_id', 'nunique'),
        gross_orders=('order_id', 'nunique'),
        accepted_orders=('is_accepted', 'nunique'),
        net_orders=('is_dropped', 'nunique'),
        cobrm_orders=('is_cobrm', 'nunique'),
        cobra_orders=('is_cobra', 'nunique'),
        ocara_orders=('is_ocara', 'nunique'),
        expired_orders=('is_expired', 'nunique'),

        # COBRA TTC
        mean_cobra_ttc=('cobra_ttc', lambda x: np.round(np.mean(x),2)),
        p25_cobra_ttc=('cobra_ttc', lambda x: safe_percentile(x, 25)),
        p50_cobra_ttc=('cobra_ttc', lambda x: safe_percentile(x, 50)),
        p75_cobra_ttc=('cobra_ttc', lambda x: safe_percentile(x, 75)),
        p90_cobra_ttc=('cobra_ttc', lambda x: safe_percentile(x, 90)),

        # OCARA TTC
        mean_ocara_ttc=('ocara_ttc', lambda x: np.round(np.mean(x),2)),
        p25_ocara_ttc=('ocara_ttc', lambda x: safe_percentile(x, 25)),
        p50_ocara_ttc=('ocara_ttc', lambda x: safe_percentile(x, 50)),
        p75_ocara_ttc=('ocara_ttc', lambda x: safe_percentile(x, 75)),
        p90_ocara_ttc=('ocara_ttc', lambda x: safe_percentile(x, 90)),

        # TTA
        mean_tta=('tta', lambda x: np.round(np.mean(x),2)),
        p25_tta=('tta', lambda x: safe_percentile(x, 25)),
        p50_tta=('tta', lambda x: safe_percentile(x, 50)),
        p75_tta=('tta', lambda x: safe_percentile(x, 75)),
        p90_tta=('tta', lambda x: safe_percentile(x, 90)),

        # ETA
        mean_eta=('eta', lambda x: np.round(np.mean(x),2)),
        p25_eta=('eta', lambda x: safe_percentile(x, 25)),
        p50_eta=('eta', lambda x: safe_percentile(x, 50)),
        p75_eta=('eta', lambda x: safe_percentile(x, 75)),
        p90_eta=('eta', lambda x: safe_percentile(x, 90)),

        avg_mr = ('map_riders_count', lambda x: np.round(np.mean(x),2)),
        p50_mr = ('map_riders_count', lambda x: safe_percentile(x, 50)),

        avg_fm = ('accept_to_pickup_distance', lambda x: np.round(np.mean(x),2)),
        p50_fm = ('accept_to_pickup_distance', lambda x: safe_percentile(x, 50)),

        # ppkm
        avg_ppkm = ('ppkm', lambda x: np.round(np.mean(x),2)),
        p50_ppkm = ('ppkm', lambda x: safe_percentile(x, 50)),

        accepted_pings = ('rider_accepted_pings', 'sum'),
        rejected_pings = ('rider_rejected_pings', 'sum'),
        busy_pings = ('rider_busy_pings', 'sum'),

        posbid_orders=('is_positive_bid', 'nunique'),
        negbid_orders=('is_negative_bid', 'nunique'),
        zerobid_orders=('is_zero_bid', 'nunique'),

        mean_posbid=('pos_amount_bid', lambda x: np.round(np.mean(x),2)),
        p25_posbid=('pos_amount_bid', lambda x: safe_percentile(x, 25)),
        p50_posbid=('pos_amount_bid', lambda x: safe_percentile(x, 50)),
        p75_posbid=('pos_amount_bid', lambda x: safe_percentile(x, 75)),
        p90_posbid=('pos_amount_bid', lambda x: safe_percentile(x, 90)),   

        mean_negbid=('neg_amount_bid', lambda x: np.round(np.mean(x),2)),
        p25_negbid=('neg_amount_bid', lambda x: safe_percentile(x, 25)),
        p50_negbid=('neg_amount_bid', lambda x: safe_percentile(x, 50)),
        p75_negbid=('neg_amount_bid', lambda x: safe_percentile(x, 75)),
        p90_negbid=('neg_amount_bid', lambda x: safe_percentile(x, 90)) 
    )

    agg_df['gross_order%'] = (
        agg_df['gross_orders'] /
        agg_df.groupby(group_cols[0])['gross_orders'].transform('sum') * 100
    ).round(2)

    # Add post-calculated metrics
    agg_df['AOR'] = (agg_df['accepted_orders'] * 100 / agg_df['gross_orders']).round(2)
    agg_df['G2N'] = (agg_df['net_orders'] * 100 / agg_df['gross_orders']).round(2)
    agg_df['COBRM'] = (agg_df['cobrm_orders'] * 100 / agg_df['gross_orders']).round(2)
    agg_df['COBRA'] = (agg_df['cobra_orders'] * 100 / agg_df['gross_orders']).round(2)
    agg_df['OCARA'] = (agg_df['ocara_orders'] * 100 / agg_df['gross_orders']).round(2)
    agg_df['expired'] = (agg_df['expired_orders'] * 100 / agg_df['gross_orders']).round(2)
    
    agg_df['bid_orders'] = (agg_df['posbid_orders'] + agg_df['negbid_orders'])

    # agg_df['bid(out of gross)%'] = (agg_df['bid_orders'] * 100 / agg_df['gross_orders']).round(2)
    agg_df['posbid%'] = (agg_df['posbid_orders'] * 100 / agg_df['gross_orders']).round(2)
    agg_df['negbid%'] = (agg_df['negbid_orders'] * 100 / agg_df['gross_orders']).round(2)
    agg_df['zerobid%'] = (agg_df['zerobid_orders'] * 100 / agg_df['gross_orders']).round(2)


    # Total pings
    agg_df['total_pings'] = (
        agg_df['accepted_pings'] +
        agg_df['rejected_pings'] +
        agg_df['busy_pings']
    )

    agg_df['APR'] = np.where(agg_df['total_pings'] > 0,
                             agg_df['accepted_pings'] / agg_df['total_pings'], np.nan).round(2)

    agg_df['DAPR'] = np.where(agg_df['accepted_pings'] > 0,
                              agg_df['net_orders'] / agg_df['accepted_pings'], np.nan).round(2)

    agg_df['DPR'] = np.where(agg_df['total_pings'] > 0,
                             agg_df['net_orders'] / agg_df['total_pings'], np.nan).round(2)

    agg_df['ME'] = np.where(agg_df['net_orders'] > 0,
                            agg_df['total_pings'] / agg_df['net_orders'], np.nan).round(2)

    agg_df['rejected_PR'] = np.where(agg_df['total_pings'] > 0,
                                     agg_df['rejected_pings'] / agg_df['total_pings'], np.nan).round(2)

    agg_df['busy_PR'] = np.where(agg_df['total_pings'] > 0,
                                 agg_df['busy_pings'] / agg_df['total_pings'], np.nan).round(2)


    cols_order = group_cols+['gross_customers', 'gross_order%', 'gross_orders', 'accepted_orders',
       'net_orders', 'cobrm_orders', 'cobra_orders', 'ocara_orders',
       'expired_orders', 'AOR', 'G2N', 'COBRM', 'COBRA', 'OCARA', 'expired',
        'avg_mr', 'p50_mr', 'avg_fm', 'p50_fm',
        'avg_ppkm', 'p50_ppkm', 'mean_cobra_ttc', 'p25_cobra_ttc', 'p50_cobra_ttc',
       'p75_cobra_ttc', 'p90_cobra_ttc', 'mean_ocara_ttc', 'p25_ocara_ttc',
       'p50_ocara_ttc', 'p75_ocara_ttc', 'p90_ocara_ttc', 'mean_tta',
       'p25_tta', 'p50_tta', 'p75_tta', 'p90_tta', 'mean_eta', 'p25_eta',
       'p50_eta', 'p75_eta', 'p90_eta', 'posbid_orders', 'negbid_orders', 'zerobid_orders',
        'bid_orders', 
                             # 'bid(out of gross)%',
                             'posbid%' , 'negbid%','zerobid%', 'mean_posbid', 'p25_posbid', 'p50_posbid', 'p75_posbid', 'p90_posbid',
        'accepted_pings', 'rejected_pings',
       'busy_pings', 'total_pings', 'APR', 'DAPR', 'DPR', 'ME',
       'rejected_PR', 'busy_PR']

    return agg_df.reset_index()[cols_order]
