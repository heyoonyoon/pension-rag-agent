'use client';

import {
  ArrowUp,
  BookOpenText,
  CheckCircle2,
  CircleAlert,
  Database,
  Landmark,
  LoaderCircle,
  Route,
  ShieldCheck,
} from 'lucide-react';
import { KeyboardEvent, SyntheticEvent, useCallback, useEffect, useMemo, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';

type AnswerResponse = {
  question_id: string;
  question: string;
  retrieved_context: string;
  think_trace: string;
  answer: string;
};

type SourceChunk = {
  id: string;
  title: string;
  content: string;
};

const exampleQuestions = [
  'IRP는 아무 때나 중도인출할 수 있나요? 중도인출하면 세금은 어떻게 되나요?',
  '퇴직연금 DB형과 DC형은 무엇이 다른가요?',
  'IRP와 연금저축의 세액공제 한도는 어떻게 되나요?',
];

function parseStatus(trace: string) {
  return trace.match(/status=([^;]+)/)?.[1] ?? 'UNKNOWN';
}

function parseSources(context: string): SourceChunk[] {
  if (!context.trim()) return [];
  return context
    .split(/\n\n(?=\[[^\]]+\])/)
    .map((block) => {
      const [firstLine = '', ...rest] = block.split('\n');
      const match = firstLine.match(/^\[([^\]]+)\]\s*(.*)$/);
      return {
        id: match?.[1] ?? 'unknown',
        title: match?.[2] ?? '검색 근거',
        content: rest.join('\n').trim(),
      };
    });
}

export default function Home() {
  const [question, setQuestion] = useState(exampleQuestions[0]);
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [serviceReady, setServiceReady] = useState<boolean | null>(null);

  const sources = useMemo(
    () => parseSources(result?.retrieved_context ?? ''),
    [result?.retrieved_context],
  );
  const status = result ? parseStatus(result.think_trace) : null;

  const requestAnswer = useCallback(async (normalized: string) => {
    setLoading(true);
    setError('');
    const params = new URLSearchParams({
      question_id: `local-${Date.now()}`,
      question: normalized,
    });

    try {
      const response = await fetch(`/api/answer?${params.toString()}`);
      if (!response.ok) {
        const payload: unknown = await response.json().catch(() => null);
        const detail =
          typeof payload === 'object' &&
          payload !== null &&
          'detail' in payload &&
          typeof payload.detail === 'string'
            ? payload.detail
            : `요청 실패 (${response.status})`;
        throw new Error(detail);
      }
      const nextResult = (await response.json()) as AnswerResponse;
      setResult(nextResult);
      return {
        question_id: nextResult.question_id,
        answer: nextResult.answer,
        status: parseStatus(nextResult.think_trace),
      };
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : '답변 요청에 실패했습니다.';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetch('/api/health/ready')
      .then((response) => response.json())
      .then((data: unknown) =>
        setServiceReady(
          typeof data === 'object' && data !== null && 'status' in data && data.status === 'ok',
        ),
      )
      .catch(() => setServiceReady(false));
  }, []);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    void Promise.resolve(
      context.registerTool(
        {
          name: 'ask_pension_system_question',
          title: '연금제도 질문하기',
          description: '연금제도 질문을 제출하고 검증된 답변을 현재 화면에 표시합니다.',
          inputSchema: {
            type: 'object',
            properties: {
              question: { type: 'string', minLength: 1, maxLength: 1000 },
            },
            required: ['question'],
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: true },
          async execute(input: unknown) {
            if (
              typeof input !== 'object' ||
              input === null ||
              !('question' in input) ||
              typeof input.question !== 'string' ||
              !input.question.trim()
            ) {
              throw new Error('question은 비어 있지 않은 문자열이어야 합니다.');
            }
            const nextQuestion = input.question.trim();
            setQuestion(nextQuestion);
            return requestAnswer(nextQuestion);
          },
        },
        { signal: lifecycle.signal },
      ),
    ).catch(() => undefined);
    return () => lifecycle.abort();
  }, [requestAnswer]);

  async function ask(event?: SyntheticEvent<HTMLFormElement>) {
    event?.preventDefault();
    const normalized = question.trim();
    if (!normalized || loading) return;

    await requestAnswer(normalized).catch(() => undefined);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      void ask();
    }
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="border-b border-white/10 bg-[#08111f] text-white">
        <div className="mx-auto flex min-h-18 max-w-[1520px] items-center justify-between gap-4 px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl bg-cyan-300 text-[#07111e] shadow-[0_0_30px_rgba(103,232,249,.2)]">
              <Landmark className="size-5" aria-hidden="true" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-[-0.02em]">연금제도 Agent</h1>
              <p className="text-sm text-slate-400">로컬 근거 검증 콘솔</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-sm">
            <span
              className={`size-2 rounded-full ${
                serviceReady === null
                  ? 'bg-amber-300'
                  : serviceReady
                    ? 'bg-emerald-400'
                    : 'bg-rose-400'
              }`}
            />
            <span className="text-slate-300">
              {serviceReady === null
                ? '연결 확인 중'
                : serviceReady
                  ? '시스템 정상'
                  : 'API 연결 필요'}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1520px] gap-5 p-5 lg:grid-cols-[minmax(0,1.45fr)_minmax(380px,.75fr)] lg:p-8">
        <section className="flex min-h-[calc(100vh-136px)] flex-col overflow-hidden rounded-2xl border bg-card shadow-sm">
          <div className="border-b px-5 py-4 sm:px-7">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-cyan-700">질문 테스트</p>
                <h2 className="mt-1 text-xl font-bold tracking-[-0.02em] sm:text-2xl">
                  연금제도에 관해 물어보세요
                </h2>
              </div>
              <Badge variant="outline" className="h-7 border-cyan-200 bg-cyan-50 px-3 text-cyan-800">
                raw_markdown 57개 문서만 사용
              </Badge>
            </div>
          </div>

          <div className="flex flex-1 flex-col justify-between gap-6 overflow-y-auto bg-[linear-gradient(180deg,#f8fbfd_0%,#fff_42%)] p-5 sm:p-7">
            <div aria-live="polite">
              {!result && !loading && !error ? (
                <div className="mx-auto flex max-w-xl flex-col items-center py-12 text-center sm:py-20">
                  <div className="grid size-16 place-items-center rounded-2xl border border-cyan-100 bg-cyan-50 text-cyan-700">
                    <ShieldCheck className="size-8" aria-hidden="true" />
                  </div>
                  <h3 className="mt-5 text-xl font-bold">근거가 검증된 답변만 표시합니다</h3>
                  <p className="mt-2 max-w-md text-base leading-7 text-muted-foreground">
                    온톨로지와 하이브리드 검색으로 원문을 찾고, 인용과 숫자가 실제 근거에
                    존재하는지 확인한 뒤 답변합니다.
                  </p>
                </div>
              ) : null}

              {loading ? (
                <div className="mx-auto flex max-w-md flex-col items-center py-16 text-center">
                  <LoaderCircle className="size-9 animate-spin text-cyan-700" aria-hidden="true" />
                  <h3 className="mt-5 text-lg font-bold">근거를 찾고 검증하고 있습니다</h3>
                  <p className="mt-2 text-sm text-muted-foreground">
                    질문에 따라 약 10~30초 정도 걸릴 수 있습니다.
                  </p>
                </div>
              ) : null}

              {error ? (
                <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-rose-900">
                  <div className="flex gap-3">
                    <CircleAlert className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
                    <div>
                      <p className="font-semibold">답변을 불러오지 못했습니다</p>
                      <p className="mt-1 text-sm">{error}</p>
                    </div>
                  </div>
                </div>
              ) : null}

              {result && !loading ? (
                <div className="space-y-5">
                  <div className="flex justify-end">
                    <div className="max-w-[88%] rounded-2xl rounded-br-md bg-[#10213a] px-5 py-4 text-[15px] leading-7 text-white shadow-sm">
                      {result.question}
                    </div>
                  </div>
                  <article className="max-w-[94%] rounded-2xl rounded-tl-md border bg-white p-5 shadow-sm sm:p-6">
                    <div className="mb-4 flex flex-wrap items-center gap-2">
                      <div className="grid size-8 place-items-center rounded-lg bg-cyan-100 text-cyan-800">
                        <Landmark className="size-4" aria-hidden="true" />
                      </div>
                      <span className="font-semibold">연금제도 Agent</span>
                      <Badge
                        variant={status === 'ANSWERED' ? 'secondary' : 'outline'}
                        className={
                          status === 'ANSWERED'
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'border-amber-200 bg-amber-50 text-amber-800'
                        }
                      >
                        {status === 'ANSWERED'
                          ? '검증 통과'
                          : status === 'OUT_OF_SCOPE'
                            ? '범위 밖 질문'
                            : status === 'INSUFFICIENT_EVIDENCE'
                              ? '근거 부족'
                              : status}
                      </Badge>
                    </div>
                    <div className="whitespace-pre-wrap text-base leading-8 text-slate-800">
                      {result.answer}
                    </div>
                  </article>
                </div>
              ) : null}
            </div>

            <form onSubmit={ask} className="sticky bottom-0 rounded-2xl border bg-white p-3 shadow-[0_18px_55px_rgba(15,23,42,.12)]">
              <label htmlFor="question" className="sr-only">
                연금제도 질문
              </label>
              <Textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                placeholder="예: IRP 중도인출 조건과 세금이 궁금해요"
                className="min-h-24 resize-none border-0 bg-transparent px-2 py-2 text-base leading-7 shadow-none focus-visible:ring-0"
              />
              <div className="mt-2 flex items-center justify-between gap-3 border-t px-1 pt-3">
                <span className="hidden text-xs text-muted-foreground sm:inline">
                  ⌘ Enter 또는 Ctrl Enter로 전송
                </span>
                <Button
                  type="submit"
                  size="lg"
                  disabled={loading || !question.trim()}
                  className="ml-auto h-10 rounded-xl bg-cyan-300 px-4 text-[#07111e] hover:bg-cyan-200"
                >
                  {loading ? '검증 중' : '질문 보내기'}
                  {loading ? (
                    <LoaderCircle className="animate-spin" aria-hidden="true" />
                  ) : (
                    <ArrowUp aria-hidden="true" />
                  )}
                </Button>
              </div>
            </form>
          </div>
        </section>

        <aside className="space-y-5">
          <section className="rounded-2xl border bg-card p-5 shadow-sm">
            <div className="flex items-center gap-2">
              <BookOpenText className="size-5 text-cyan-700" aria-hidden="true" />
              <h2 className="font-bold">빠른 테스트</h2>
            </div>
            <div className="mt-4 grid gap-2">
              {exampleQuestions.map((example, index) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => setQuestion(example)}
                  disabled={loading}
                  className="group flex min-h-12 items-start gap-3 rounded-xl border bg-white px-4 py-3 text-left text-sm leading-6 transition hover:border-cyan-300 hover:bg-cyan-50 disabled:opacity-50"
                >
                  <span className="mt-0.5 font-mono text-xs font-bold text-cyan-700">
                    0{index + 1}
                  </span>
                  <span>{example}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="overflow-hidden rounded-2xl border bg-card shadow-sm">
            <div className="flex items-center justify-between border-b px-5 py-4">
              <div className="flex items-center gap-2">
                <Database className="size-5 text-cyan-700" aria-hidden="true" />
                <h2 className="font-bold">검증 상세</h2>
              </div>
              {result ? (
                <span className="text-xs text-muted-foreground">{sources.length}개 근거 청크</span>
              ) : null}
            </div>

            {result ? (
              <Tabs defaultValue="sources" className="min-h-[330px]">
                <TabsList variant="line" className="mx-5 mt-3 w-[calc(100%-2.5rem)] justify-start">
                  <TabsTrigger value="sources" className="flex-none px-3">검색 근거</TabsTrigger>
                  <TabsTrigger value="trace" className="flex-none px-3">실행 기록</TabsTrigger>
                </TabsList>
                <TabsContent value="sources" className="max-h-[520px] overflow-y-auto p-5 pt-3">
                  {sources.length ? (
                    <div className="space-y-3">
                      {sources.map((source, index) => (
                        <article key={`${source.id}-${index}`} className="rounded-xl border bg-slate-50 p-4">
                          <div className="flex items-start justify-between gap-2">
                            <h3 className="text-sm font-bold leading-6">{source.title}</h3>
                            <Badge variant="outline" className="max-w-40 shrink-0 font-mono text-[11px]">
                              {source.id}
                            </Badge>
                          </div>
                          <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-600">
                            {source.content}
                          </p>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <p className="py-10 text-center text-sm text-muted-foreground">
                      이 응답에는 검색 근거가 없습니다.
                    </p>
                  )}
                </TabsContent>
                <TabsContent value="trace" className="p-5 pt-3">
                  <div className="rounded-xl bg-[#08111f] p-4 text-slate-200">
                    <div className="mb-3 flex items-center gap-2 text-cyan-300">
                      <Route className="size-4" aria-hidden="true" />
                      <span className="text-sm font-semibold">LangGraph 실행 단계</span>
                    </div>
                    <p className="whitespace-pre-wrap break-words font-mono text-[13px] leading-7">
                      {result.think_trace.split(' → ').join('\n→ ')}
                    </p>
                  </div>
                </TabsContent>
              </Tabs>
            ) : (
              <div className="flex min-h-[300px] flex-col items-center justify-center px-8 text-center text-muted-foreground">
                <CheckCircle2 className="size-8 text-slate-300" aria-hidden="true" />
                <p className="mt-3 text-sm leading-6">
                  질문 후 검색된 원문 청크와<br />검증 실행 단계를 확인할 수 있습니다.
                </p>
              </div>
            )}
          </section>

          <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-950">
            현재는 <strong>연금제도 질문만</strong> 답변합니다. 펀드·상품 선택이나 수익률 기반
            추천은 범위 밖으로 처리됩니다.
          </div>
        </aside>
      </div>
    </main>
  );
}
