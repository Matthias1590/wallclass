import time
import requests
import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from wallets import customer_addrs, exchange_addrs, bot_addrs
from typing import Optional

API_KEY = "SMWQ1AEZWM7MEPQ4GYW6WXBVWGCQM6MPII"

def _get_data(address: str, action: str) -> dict:
    print(f"Fetching {action} for {address}")
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=account&action={action}&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}"
    res = requests.get(url)
    res.raise_for_status()
    return res.json()

def get_data(address: str, action: str) -> dict:
    os.makedirs("data", exist_ok=True)
    path = f"data/{address}_{action}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    data = _get_data(address, action)
    with open(path, "w") as f:
        json.dump(data, f)
    return data

def calculate_entropy(arr: np.ndarray) -> float:
    """Calculate entropy of a distribution using NumPy."""
    if arr.size == 0:
        return 0.0
    
    total = np.sum(arr)
    if total == 0:
        return 0.0
        
    probs = arr / total
    non_zero = probs > 0
    if not np.any(non_zero):
        return 0.0
        
    return float(-np.sum(probs[non_zero] * np.log(probs[non_zero])))

exchange_addrs_ls = set([addr.lower() for addr in exchange_addrs])

def get_features(addr: str) -> Optional[dict]:
    txs = get_data(addr, "txlist").get("result", [])
    # print("Calculating features for", addr)

    if "Max calls per sec rate limit" in txs:
        print("Rate limited")
        time.sleep(1)
        return get_features(addr)
    if not txs:
        return None

    # Convert relevant transaction data to NumPy arrays for faster processing
    try:
        timestamps = np.array([int(tx["timeStamp"]) for tx in txs])
    except:
        print(addr)
        print(txs)
        exit()
    start_ts = np.min(timestamps)
    end_ts = np.max(timestamps)
    lifetime_s = end_ts - start_ts
    lifetime_days = lifetime_s / (24 * 60 * 60)
    
    if not lifetime_days:
        return None
        
    # Calculate time intervals using vectorized operations
    tx_intervals = timestamps[1:] - timestamps[:-1]
    
    # Create boolean masks for filtering
    addr_lower = addr.lower()
    is_incoming = np.array([tx["to"].lower() == addr_lower for tx in txs])
    is_outgoing = np.array([tx["from"].lower() == addr_lower for tx in txs])
    
    is_from_exchange = np.array([tx["from"].lower() in exchange_addrs_ls for tx in txs])
    is_to_exchange = np.array([tx["to"].lower() in exchange_addrs_ls for tx in txs])
    
    # Filter and calculate tx values
    tx_values = np.array([float(tx["value"]) if tx["value"].isdigit() else 0 for tx in txs])
    valid_values = tx_values > 0  # Mask for valid values
    
    # Calculate volume in/out using masks
    volume_in = np.sum(tx_values * is_incoming)
    volume_out = np.sum(tx_values * is_outgoing)
    
    # Get unique counterparties
    from_addresses = set(tx["from"].lower() for tx in txs)
    to_addresses = set(tx["to"].lower() for tx in txs)
    
    # Get counterparty reuse metric - simplified and fixed
    counterparty_reuse = 0.0
    if txs:
        # Calculate unique counterparties for incoming and outgoing transactions
        incoming_counterparties = set(tx["from"].lower() for tx in txs if tx["to"].lower() == addr_lower)
        outgoing_counterparties = set(tx["to"].lower() for tx in txs if tx["from"].lower() == addr_lower)
        
        # Calculate reuse ratio: how many unique counterparties vs total transactions
        total_unique_counterparties = len(incoming_counterparties | outgoing_counterparties)
        if total_unique_counterparties > 0:
            counterparty_reuse = total_unique_counterparties / len(txs)
        else:
            counterparty_reuse = 0.0
    
    features = {
        "tx_count_total": len(txs),
        "tx_count_incoming": np.sum(is_incoming),
        "tx_count_outgoing": np.sum(is_outgoing),
        "txs_per_day": len(txs) / lifetime_days,
        "first_tx_time": float(start_ts),
        "last_tx_time": float(end_ts),
        "mean_tx_interval": float(np.mean(tx_intervals)) if len(tx_intervals) > 0 else 0.0,
        "std_tx_interval": float(np.std(tx_intervals)) if len(tx_intervals) > 0 else 0.0,
        "tx_interval_entropy": calculate_entropy(tx_intervals) if len(tx_intervals) > 0 else 0.0,
        "counterparties_count": len(from_addresses | to_addresses),
        "in_out_ratio": float(np.sum(is_outgoing) / max(1, np.sum(is_incoming))),
        "exchange_counterparty_ratio": float(np.sum(is_from_exchange) / max(1, np.sum(is_to_exchange))),
        "avg_counterparty_reuse": counterparty_reuse,
        "mean_tx_value": float(np.mean(tx_values[valid_values])) if np.any(valid_values) else 0.0,
        "median_tx_value": float(np.median(tx_values[valid_values])) if np.any(valid_values) else 0.0,
        "std_tx_value": float(np.std(tx_values[valid_values])) if np.any(valid_values) else 0.0,
        "max_tx_value": float(np.max(tx_values[valid_values])) if np.any(valid_values) else 0.0,
        "total_volume_in": float(volume_in),
        "total_volume_out": float(volume_out),
    }

    # print("Done")
    return features

def main() -> None:
    addrs_all = []
    for addr in bot_addrs:
        addrs_all.append((addr, "bot"))
    for addr in customer_addrs:
        addrs_all.append((addr, "customer"))
    for addr in exchange_addrs:
        addrs_all.append((addr, "exchange"))

    features_all = []
    for i in tqdm(range(len(addrs_all))):
        addr, addr_type = addrs_all[i]
        feature = get_features(addr)

        if feature is not None:
            feature["label"] = addr_type
            features_all.append(feature)

    # Save to file
    features_df = pd.DataFrame(features_all)
    features_df.to_csv("features_all.csv", index=True)

if __name__ == "__main__":
    main()
