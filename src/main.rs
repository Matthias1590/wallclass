mod wallet_classifier;
mod etherscan;

use ethers::prelude::*;
use crate::wallet_classifier::WalletClassifier;
use crate::etherscan;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let etherscan_client = etherscan::Client::new("SMWQ1AEZWM7MEPQ4GYW6WXBVWGCQM6MPII");

    // let provider = Provider::<Http>::try_from("https://mainnet.infura.io/v3/ae660ba403ee422a80f1855e385ec11c")?;
    // let wallet_classifier = WalletClassifier::new(provider);

    let address = "0xf60c2ea62edbfe808163751dd0d8693dcb30019c";

    // let class = wallet_classifier.classify(address).await?;

    // println!("{:?}", class);

    let bal = etherscan::account::get_balance(Config::new("account".into(), "balance".into(), address.into(), "latest".into(), "SMWQ1AEZWM7MEPQ4GYW6WXBVWGCQM6MPII".into())).await?;
    println!("Balance: {}", bal);

    Ok(())
}
