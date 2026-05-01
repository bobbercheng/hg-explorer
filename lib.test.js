const {
  formatNum,
  formatParams,
  getModelCategory,
  getBaseModelId,
  getParamCount,
  getInferenceProviders,
  buildFetchUrl,
  filterModels,
  computeStats,
  buildBaseModelTree,
} = require('./lib');

// --- formatNum ---
describe('formatNum', () => {
  test('formats billions', () => {
    expect(formatNum(1e9)).toBe('1.0B');
    expect(formatNum(2.5e9)).toBe('2.5B');
  });
  test('formats millions', () => {
    expect(formatNum(1e6)).toBe('1.0M');
    expect(formatNum(19788680)).toBe('19.8M');
  });
  test('formats thousands', () => {
    expect(formatNum(1000)).toBe('1.0K');
    expect(formatNum(5432)).toBe('5.4K');
  });
  test('formats small numbers as-is', () => {
    expect(formatNum(0)).toBe('0');
    expect(formatNum(999)).toBe('999');
  });
});

// --- formatParams ---
describe('formatParams', () => {
  test('returns Unknown for falsy values', () => {
    expect(formatParams(0)).toBe('Unknown');
    expect(formatParams(null)).toBe('Unknown');
    expect(formatParams(undefined)).toBe('Unknown');
  });
  test('formats billions', () => {
    expect(formatParams(7e9)).toBe('7.0B');
    expect(formatParams(70553706496)).toBe('70.6B');
  });
  test('formats millions', () => {
    expect(formatParams(137e6)).toBe('137M');
    expect(formatParams(751632384)).toBe('752M');
  });
  test('formats thousands', () => {
    expect(formatParams(500000)).toBe('500K');
  });
});

// --- getModelCategory ---
describe('getModelCategory', () => {
  test('returns base when no base_model tags', () => {
    expect(getModelCategory({ tags: ['transformers', 'text-generation'] })).toBe('base');
  });

  test('returns base for empty/missing tags', () => {
    expect(getModelCategory({})).toBe('base');
    expect(getModelCategory({ tags: [] })).toBe('base');
  });

  test('returns finetune for base_model:finetune: tag', () => {
    expect(getModelCategory({
      tags: ['transformers', 'base_model:finetune:meta-llama/Llama-3.1-8B']
    })).toBe('finetune');
  });

  test('returns finetune for plain base_model: tag', () => {
    expect(getModelCategory({
      tags: ['base_model:meta-llama/Llama-3.1-8B']
    })).toBe('finetune');
  });

  test('returns adapter for base_model:adapter: tag', () => {
    expect(getModelCategory({
      tags: ['base_model:adapter:Qwen/Qwen3-8B']
    })).toBe('adapter');
  });

  test('returns adapter for lora tag', () => {
    expect(getModelCategory({ tags: ['lora'] })).toBe('adapter');
  });

  test('returns adapter for peft tag', () => {
    expect(getModelCategory({ tags: ['peft'] })).toBe('adapter');
  });

  test('returns quantized for base_model:quantized: tag', () => {
    expect(getModelCategory({
      tags: ['base_model:quantized:Qwen/Qwen3-35B']
    })).toBe('quantized');
  });

  test('adapter takes priority over finetune', () => {
    expect(getModelCategory({
      tags: ['base_model:finetune:Qwen/Qwen3-8B', 'lora', 'peft']
    })).toBe('adapter');
  });

  test('quantized takes priority over finetune', () => {
    expect(getModelCategory({
      tags: ['base_model:quantized:Qwen/Qwen3-35B', 'base_model:Qwen/Qwen3-35B']
    })).toBe('quantized');
  });
});

// --- getBaseModelId ---
describe('getBaseModelId', () => {
  test('returns null for base models with no base_model tag', () => {
    expect(getBaseModelId({ tags: ['transformers'] })).toBeNull();
    expect(getBaseModelId({})).toBeNull();
  });

  test('extracts from base_model:finetune: tag', () => {
    expect(getBaseModelId({
      tags: ['base_model:finetune:meta-llama/Llama-3.1-8B']
    })).toBe('meta-llama/Llama-3.1-8B');
  });

  test('extracts from base_model:adapter: tag', () => {
    expect(getBaseModelId({
      tags: ['base_model:adapter:Qwen/Qwen3-8B']
    })).toBe('Qwen/Qwen3-8B');
  });

  test('extracts from base_model:quantized: tag', () => {
    expect(getBaseModelId({
      tags: ['base_model:quantized:Qwen/Qwen3-35B']
    })).toBe('Qwen/Qwen3-35B');
  });

  test('extracts from plain base_model: tag', () => {
    expect(getBaseModelId({
      tags: ['base_model:meta-llama/Llama-3.1-70B']
    })).toBe('meta-llama/Llama-3.1-70B');
  });

  test('prioritizes finetune prefix over plain base_model', () => {
    // The first matching tag wins based on iteration order
    const result = getBaseModelId({
      tags: ['base_model:finetune:A/B', 'base_model:C/D']
    });
    expect(result).toBe('A/B');
  });
});

// --- getParamCount ---
describe('getParamCount', () => {
  test('returns safetensors total when available', () => {
    expect(getParamCount({
      id: 'Qwen/Qwen3-0.6B',
      safetensors: { total: 751632384 }
    })).toBe(751632384);
  });

  test('parses B from model name when no safetensors', () => {
    expect(getParamCount({ id: 'meta-llama/Llama-3.1-70B-Instruct' })).toBe(70e9);
    expect(getParamCount({ id: 'Qwen/Qwen3-0.6B' })).toBe(0.6e9);
  });

  test('parses M from model name', () => {
    expect(getParamCount({ id: 'some-org/model-125M' })).toBe(125e6);
  });

  test('returns null when no size info', () => {
    expect(getParamCount({ id: 'openai-community/gpt2' })).toBeNull();
  });

  test('safetensors takes priority over name parsing', () => {
    expect(getParamCount({
      id: 'Qwen/Qwen3-0.6B',
      safetensors: { total: 999 }
    })).toBe(999);
  });
});

// --- getInferenceProviders ---
describe('getInferenceProviders', () => {
  test('returns empty array for no mapping', () => {
    expect(getInferenceProviders({})).toEqual([]);
    expect(getInferenceProviders({ inferenceProviderMapping: null })).toEqual([]);
  });

  test('returns empty array for non-array mapping', () => {
    expect(getInferenceProviders({ inferenceProviderMapping: {} })).toEqual([]);
    expect(getInferenceProviders({ inferenceProviderMapping: 'invalid' })).toEqual([]);
  });

  test('returns empty array for empty array', () => {
    expect(getInferenceProviders({ inferenceProviderMapping: [] })).toEqual([]);
  });

  test('returns only live providers', () => {
    const model = {
      inferenceProviderMapping: [
        { provider: 'together', status: 'live' },
        { provider: 'featherless-ai', status: 'live' },
        { provider: 'staging-provider', status: 'staging' },
      ]
    };
    expect(getInferenceProviders(model)).toEqual(['together', 'featherless-ai']);
  });

  test('filters out non-live providers', () => {
    const model = {
      inferenceProviderMapping: [
        { provider: 'broken', status: 'error' },
        { provider: 'pending', status: 'pending' },
      ]
    };
    expect(getInferenceProviders(model)).toEqual([]);
  });
});

// --- buildFetchUrl ---
describe('buildFetchUrl', () => {
  test('constructs correct URL with all parameters', () => {
    const url = buildFetchUrl('https://huggingface.co/api/models', 'text-generation', 100, 0);
    expect(url).toContain('pipeline_tag=text-generation');
    expect(url).toContain('sort=createdAt');
    expect(url).toContain('direction=-1');
    expect(url).toContain('limit=100');
    expect(url).toContain('offset=0');
    expect(url).toContain('expand[]=safetensors');
    expect(url).toContain('expand[]=inferenceProviderMapping');
    expect(url).toContain('expand[]=downloads');
    expect(url).toContain('expand[]=createdAt');
    expect(url).toContain('expand[]=tags');
  });

  test('uses provided offset', () => {
    const url = buildFetchUrl('https://huggingface.co/api/models', 'text-generation', 50, 200);
    expect(url).toContain('limit=50');
    expect(url).toContain('offset=200');
  });

  test('supports custom sort parameter', () => {
    const url = buildFetchUrl('https://huggingface.co/api/models', 'text-generation', 100, 0, 'downloads');
    expect(url).toContain('sort=downloads');
  });

  test('defaults to createdAt sort', () => {
    const url = buildFetchUrl('https://huggingface.co/api/models', 'text-generation', 100, 0);
    expect(url).toContain('sort=createdAt');
  });
});

// --- filterModels ---
describe('filterModels', () => {
  const cutoff = new Date('2026-02-01T00:00:00Z');

  test('accepts models after cutoff with enough downloads', () => {
    const data = [
      { createdAt: '2026-03-15T00:00:00Z', downloads: 5000 },
      { createdAt: '2026-02-10T00:00:00Z', downloads: 2000 },
    ];
    const result = filterModels(data, cutoff, 1000);
    expect(result.accepted).toHaveLength(2);
    expect(result.reachedCutoff).toBe(false);
  });

  test('rejects models below download threshold', () => {
    const data = [
      { createdAt: '2026-03-15T00:00:00Z', downloads: 500 },
      { createdAt: '2026-03-10T00:00:00Z', downloads: 100 },
    ];
    const result = filterModels(data, cutoff, 1000);
    expect(result.accepted).toHaveLength(0);
    expect(result.reachedCutoff).toBe(false);
  });

  test('stops at cutoff date', () => {
    const data = [
      { createdAt: '2026-03-15T00:00:00Z', downloads: 5000 },
      { createdAt: '2026-01-15T00:00:00Z', downloads: 9000 },
      { createdAt: '2026-01-01T00:00:00Z', downloads: 3000 },
    ];
    const result = filterModels(data, cutoff, 1000);
    expect(result.accepted).toHaveLength(1);
    expect(result.reachedCutoff).toBe(true);
  });

  test('handles empty data', () => {
    const result = filterModels([], cutoff, 1000);
    expect(result.accepted).toHaveLength(0);
    expect(result.reachedCutoff).toBe(false);
  });

  test('handles models with missing downloads', () => {
    const data = [
      { createdAt: '2026-03-15T00:00:00Z' },
    ];
    const result = filterModels(data, cutoff, 1000);
    expect(result.accepted).toHaveLength(0);
  });
});

// --- computeStats ---
describe('computeStats', () => {
  const models = [
    { tags: [], downloads: 1000, inferenceProviderMapping: [{ provider: 'together', status: 'live' }] },
    { tags: ['base_model:finetune:A/B'], downloads: 2000, inferenceProviderMapping: [{ provider: 'together', status: 'live' }, { provider: 'groq', status: 'live' }] },
    { tags: ['base_model:quantized:A/B'], downloads: 500, inferenceProviderMapping: [] },
    { tags: ['lora'], downloads: 300, inferenceProviderMapping: [] },
  ];

  test('counts categories correctly', () => {
    const stats = computeStats(models);
    expect(stats.categories.base).toBe(1);
    expect(stats.categories.finetune).toBe(1);
    expect(stats.categories.quantized).toBe(1);
    expect(stats.categories.adapter).toBe(1);
  });

  test('sums total downloads', () => {
    expect(computeStats(models).totalDownloads).toBe(3800);
  });

  test('counts unique providers', () => {
    expect(computeStats(models).uniqueProviders).toBe(2);
  });

  test('handles empty model list', () => {
    const stats = computeStats([]);
    expect(stats.categories.base).toBe(0);
    expect(stats.totalDownloads).toBe(0);
    expect(stats.uniqueProviders).toBe(0);
  });
});

// --- buildBaseModelTree ---
describe('buildBaseModelTree', () => {
  test('groups derivatives under their base model', () => {
    const models = [
      { id: 'org/base-model', tags: [] },
      { id: 'user/ft-1', tags: ['base_model:finetune:org/base-model'], downloads: 500 },
      { id: 'user/ft-2', tags: ['base_model:finetune:org/base-model'], downloads: 1000 },
      { id: 'user/quant-1', tags: ['base_model:quantized:org/base-model'], downloads: 200 },
    ];
    const tree = buildBaseModelTree(models);
    expect(tree).toHaveLength(1);
    expect(tree[0][0]).toBe('org/base-model');
    expect(tree[0][1].finetunes).toHaveLength(3);
  });

  test('sorts by number of derivatives descending', () => {
    const models = [
      { id: 'u/ft-a1', tags: ['base_model:finetune:org/A'] },
      { id: 'u/ft-b1', tags: ['base_model:finetune:org/B'] },
      { id: 'u/ft-b2', tags: ['base_model:finetune:org/B'] },
      { id: 'u/ft-b3', tags: ['base_model:finetune:org/B'] },
    ];
    const tree = buildBaseModelTree(models);
    expect(tree[0][0]).toBe('org/B');
    expect(tree[0][1].finetunes).toHaveLength(3);
    expect(tree[1][0]).toBe('org/A');
    expect(tree[1][1].finetunes).toHaveLength(1);
  });

  test('excludes base models from tree entries', () => {
    const models = [
      { id: 'org/base', tags: [] },
    ];
    const tree = buildBaseModelTree(models);
    expect(tree).toHaveLength(0);
  });

  test('handles empty input', () => {
    expect(buildBaseModelTree([])).toEqual([]);
  });

  test('categorizes derivatives correctly', () => {
    const models = [
      { id: 'u/ft', tags: ['base_model:finetune:org/X'] },
      { id: 'u/lora', tags: ['base_model:adapter:org/X', 'lora'] },
      { id: 'u/quant', tags: ['base_model:quantized:org/X'] },
    ];
    const tree = buildBaseModelTree(models);
    const finetunes = tree[0][1].finetunes;
    expect(finetunes.find(f => f.model.id === 'u/ft').category).toBe('finetune');
    expect(finetunes.find(f => f.model.id === 'u/lora').category).toBe('adapter');
    expect(finetunes.find(f => f.model.id === 'u/quant').category).toBe('quantized');
  });
});
