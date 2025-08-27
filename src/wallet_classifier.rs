use ethers::{prelude::*, utils::keccak256};

#[derive(Debug)]
pub enum WalletClass {
    Customer(f32),
    MevBot(f32),
    Exchange(f32),
}

pub struct WalletClassifier {
    provider: Provider<Http>,
}

impl WalletClassifier {
    pub fn new(provider: Provider<Http>) -> Self {
        Self { provider }
    }

    // todo: not pub
    async fn get_features(&self, address: Address) -> anyhow::Result<Vec<f32>> {
        // ERC20 Transfer event topic
        let transfer_sig = "Transfer(address,address,uint256)";
        let transfer_topic = H256::from(keccak256(transfer_sig));

        let filter = Filter::new()
            .topic0(transfer_topic)
            .topic1(ValueOrArray::Value(address.into())) // from
            // .or(Filter::new()
            //     .topic0(transfer_topic)
            //     .topic2(ValueOrArray::Value(address.into()))); // to
            ;

        let logs = self.provider.get_logs(&filter).await?;
        // println!("{:#?}", logs);

        Ok(vec![logs.len() as f32])
    }

    pub async fn classify(&self, address: Address) -> anyhow::Result<WalletClass> {
        let features = self.get_features(address).await?;
        Ok(WalletClass::Exchange(features[0] as f32))
    }
}
