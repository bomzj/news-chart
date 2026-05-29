"use client";

import { AggregatedNews } from "../lib/types";
import { sentimentColor, impactRadius } from "../lib/aggregate";

type Props = {
  x: number;
  y: number;
  news: AggregatedNews;
  onClose: () => void;
};

export default function NewsPopup({ x, y, news, onClose }: Props) {
  // Position popup so it doesn't overflow viewport
  const style: React.CSSProperties = {
    position: "absolute",
    left: Math.min(x, window.innerWidth - 360),
    top: Math.max(40, y - 200),
    zIndex: 50,
  };

  const sortedItems = [...news.items].sort((a, b) => b.impact - a.impact);

  return (
    <div
      style={style}
      className="w-[340px] max-h-[300px] overflow-y-auto bg-white border border-gray-200 rounded-lg shadow-lg p-3"
      onMouseLeave={onClose}
    >
      <div className="text-xs text-gray-500 mb-2 font-semibold">
        {news.items.length} news item{news.items.length > 1 ? "s" : ""}
      </div>
      <ul className="space-y-2">
        {sortedItems.map((item, idx) => {
          const color = sentimentColor(item.sentiment);
          const radius = impactRadius(item.impact);
          return (
            <li key={idx} className="flex items-start gap-2">
              <span
                className="shrink-0 rounded-full mt-1"
                style={{
                  width: radius,
                  height: radius,
                  backgroundColor: color,
                }}
              />
              <span className="text-xs text-gray-700 leading-tight">
                {item.news_summary}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
