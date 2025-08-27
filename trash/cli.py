#!/usr/bin/env python3
"""
CLI tool for wallet classification.
Usage: python3 cli.py [wallet_address]
If no address is provided, a random address will be selected.
"""

import sys
import pandas as pd
import random
from collect import get_features
from predict import predict_wallet_probabilities
from wallets import get_addresses

def get_random_address():
    """Get a random wallet address from the known addresses."""
    selected_addr = get_addresses(1)[0]
    print(f"No address provided, using random address: {selected_addr}")
    return selected_addr

def main():
    # Check command line arguments
    if len(sys.argv) > 2:
        print("Usage: python3 cli.py [wallet_address]")
        print("Example: python3 cli.py 0x742d35Cc6635C0532925a3b8D34bA2Bfe8cc5B8f")
        print("If no address is provided, a random address will be selected.")
        sys.exit(1)
    elif len(sys.argv) == 2:
        wallet_address = sys.argv[1]
        # Validate wallet address format (basic check)
        if not wallet_address.startswith('0x') or len(wallet_address) != 42:
            print("Error: Invalid Ethereum wallet address format")
            print("Address should start with '0x' and be 42 characters long")
            sys.exit(1)
    else:
        # No address provided, use a random one
        wallet_address = get_random_address()
    
    print(f"Analyzing wallet: {wallet_address}")
    print("-" * 50)
    
    try:
        # Get features for the wallet
        print("Extracting features from blockchain data...")
        features = get_features(wallet_address)
        
        if features is None:
            print("Error: Could not extract features for this wallet address")
            print("This might happen if the wallet has no transactions or invalid data")
            sys.exit(1)
        
        # Remove the 'label' column if it exists (it won't be there from get_features, but just in case)
        if 'label' in features:
            del features['label']
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        print("Features extracted successfully!")
        print(f"Total transactions: {features.get('tx_count_total', 'N/A')}")
        print(f"Transactions per day: {features.get('txs_per_day', 'N/A'):.2f}")
        print()
        
        # Get predictions
        print("Running classification models...")
        probabilities = predict_wallet_probabilities(features_df)
        
        # Convert to percentages and display results
        print("WALLET CLASSIFICATION RESULTS:")
        print("=" * 40)
        
        bot_prob = probabilities['bot_probability'].iloc[0] * 100
        customer_prob = probabilities['customer_probability'].iloc[0] * 100
        exchange_prob = probabilities['exchange_probability'].iloc[0] * 100
        
        print(f"🤖 Bot probability:      {bot_prob:6.2f}%")
        print(f"👤 Customer probability: {customer_prob:6.2f}%")
        print(f"🏦 Exchange probability: {exchange_prob:6.2f}%")
        print()
        
        # Determine most likely class
        max_prob = max(bot_prob, customer_prob, exchange_prob)
        if max_prob == bot_prob:
            predicted_class = "Bot"
            emoji = "🤖"
        elif max_prob == customer_prob:
            predicted_class = "Customer"
            emoji = "👤"
        else:
            predicted_class = "Exchange"
            emoji = "🏦"
        
        print(f"Most likely classification: {emoji} {predicted_class} ({max_prob:.2f}%)")
        
        # Add confidence assessment
        if max_prob > 80:
            confidence = "Very High"
        elif max_prob > 60:
            confidence = "High"
        elif max_prob > 40:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        print(f"Confidence level: {confidence}")

        print(f"Etherscan: https://etherscan.io/address/{wallet_address}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
