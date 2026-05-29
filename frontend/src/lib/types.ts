export type Timeframe = "1h" | "4h" | "1d" | "1w" | "1M";

export type Sentiment = "bullish" | "bearish";
export type Impact = 1 | 2 | 3;

export type NewsItem = {
  published_at: string;
  sentiment: Sentiment;
  impact: Impact;
  news_summary: string;
  confidence: number;
  predicted_by_model: string;
  price_at_ingestion: number;
};

export type Kline = {
  time: number; // unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
};

export type SentimentSlice = {
  sentiment: "bullish" | "bearish";
  count: number;
  color: string;
};

export type AggregatedNews = {
  time: number; // candle open timestamp (unix seconds)
  items: NewsItem[];
  avgSentiment: number;
  maxImpact: Impact;
  slices: SentimentSlice[];
};
