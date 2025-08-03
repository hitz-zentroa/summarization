from enum import StrEnum

class SummarizerLLM(StrEnum):
    claude = 'claude'
    command_r = 'commandr'
    gpt_4o = 'gpt4o'
    reka = 'reka'
    llama_3 = 'llama3'

class Prompt(StrEnum):
    base = 'base'
    core = 'core'
    wh = '5w1h'
    tldr = 'tldr'

SummaryConfig = StrEnum('SummaryConfig', ['-'.join((llm, prompt)) for llm in list(SummarizerLLM) for prompt in list(Prompt)])

class Criterion(StrEnum):
    coh = 'Coherence'
    con = 'Consistency'
    flu = 'Fluency'
    rel = 'Relevance'
    whs = '5W1H'

class Metric(StrEnum):
    rogue = 'rouge'
    m_rouge_we = 'm_rouge_we'
    m_bert_score = 'm_bert_score'
    bleu = 'bleu'
    chrf = 'chrf'
    m_meteor = 'm_meteor'
    cider = 'cider'
    stats = 'stats'
