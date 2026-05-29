import { NewsItem } from "./types";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export async function fetchNews(
  ticker: string,
  fromTs?: string,
  toTs?: string,
  minImpact: number = 1,
): Promise<NewsItem[]> {
  const url = new URL(`${API_URL}/api/news`);
  url.searchParams.set("ticker", ticker);
  url.searchParams.set("min_impact", String(minImpact));
  if (fromTs) url.searchParams.set("from_ts", fromTs);
  if (toTs) url.searchParams.set("to_ts", toTs);

  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`News API error: ${res.status}`);

  const data = await res.json();
  return data.items as NewsItem[];
}
