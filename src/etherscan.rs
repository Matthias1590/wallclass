pub struct Client {
    api_key: String
}

impl Client {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    fn get_url(&self, module: &str, action: &str, address: &str) -> String {
        format!(
            "https://api.etherscan.io/api?module={}&action={}&address={}&sort=desc&apikey={}",
            module, action, address, self.api_key
        )
    }

    pub async fn get_transactions(&self, address: &str) -> anyhow::Result<serde_json::Value> {
        let url = self.get_url("account", "txlist", address);
        let resp = reqwest::get(&url).await?;
        let value: serde_json::Value = serde_json::from_str(&resp.text().await?).or(Err(anyhow::anyhow!("Failed to parse JSON")))?;
        Ok(value)
    }
}
