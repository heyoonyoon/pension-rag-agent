import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: '연금제도 Agent 검증 콘솔',
  description: '온톨로지와 하이브리드 RAG 기반 연금제도 Agent의 로컬 검증 화면',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
