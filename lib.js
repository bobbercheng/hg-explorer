function formatNum(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatParams(n) {
  if (!n) return 'Unknown';
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(0) + 'M';
  return (n / 1e3).toFixed(0) + 'K';
}

function getModelCategory(model) {
  const tags = model.tags || [];
  const hasBaseModelFT = tags.some(t => t.startsWith('base_model:finetune:'));
  const hasBaseModelAdapter = tags.some(t => t.startsWith('base_model:adapter:'));
  const hasBaseModelQuant = tags.some(t => t.startsWith('base_model:quantized:'));
  const hasBaseModel = tags.some(t => t.startsWith('base_model:') && !t.startsWith('base_model:finetune:') && !t.startsWith('base_model:adapter:') && !t.startsWith('base_model:quantized:'));

  if (hasBaseModelAdapter || tags.includes('lora') || tags.includes('peft')) return 'adapter';
  if (hasBaseModelQuant) return 'quantized';
  if (hasBaseModelFT || hasBaseModel) return 'finetune';
  return 'base';
}

function getBaseModelId(model) {
  const tags = model.tags || [];
  for (const t of tags) {
    for (const prefix of ['base_model:finetune:', 'base_model:adapter:', 'base_model:quantized:', 'base_model:']) {
      if (t.startsWith(prefix)) return t.slice(prefix.length);
    }
  }
  return null;
}

function getParamCount(model) {
  if (model.safetensors && model.safetensors.total) return model.safetensors.total;
  // Try parsing from name
  const match = model.id.match(/(\d+\.?\d*)\s*[Bb]/);
  if (match) return parseFloat(match[1]) * 1e9;
  const matchM = model.id.match(/(\d+\.?\d*)\s*[Mm]/);
  if (matchM) return parseFloat(matchM[1]) * 1e6;
  return null;
}

function getInferenceProviders(model) {
  if (!model.inferenceProviderMapping || !Array.isArray(model.inferenceProviderMapping)) return [];
  return model.inferenceProviderMapping
    .filter(p => p.status === 'live')
    .map(p => p.provider);
}

function buildFetchUrl(apiBase, pipeline, limit, offset, sort = 'createdAt') {
  return `${apiBase}?pipeline_tag=${pipeline}&sort=${sort}&direction=-1&limit=${limit}&offset=${offset}&expand[]=safetensors&expand[]=inferenceProviderMapping&expand[]=downloads&expand[]=createdAt&expand[]=tags`;
}

function filterModels(data, cutoffDate, minDownloads) {
  const accepted = [];
  let reachedCutoff = false;
  for (const m of data) {
    const created = new Date(m.createdAt);
    if (created < cutoffDate) { reachedCutoff = true; break; }
    if ((m.downloads || 0) >= minDownloads) {
      accepted.push(m);
    }
  }
  return { accepted, reachedCutoff };
}

function computeStats(models) {
  const categories = { base: 0, finetune: 0, quantized: 0, adapter: 0 };
  models.forEach(m => { categories[getModelCategory(m)]++; });
  const totalDownloads = models.reduce((s, m) => s + (m.downloads || 0), 0);
  const providersSet = new Set();
  models.forEach(m => getInferenceProviders(m).forEach(p => providersSet.add(p)));
  return { categories, totalDownloads, uniqueProviders: providersSet.size };
}

function buildBaseModelTree(models) {
  const baseMap = {};
  models.forEach(m => {
    const cat = getModelCategory(m);
    if (cat !== 'base') {
      const baseId = getBaseModelId(m);
      if (baseId) {
        if (!baseMap[baseId]) baseMap[baseId] = { finetunes: [] };
        baseMap[baseId].finetunes.push({ model: m, category: cat });
      }
    }
  });
  return Object.entries(baseMap)
    .filter(([, v]) => v.finetunes.length > 0)
    .sort((a, b) => b[1].finetunes.length - a[1].finetunes.length);
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
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
  };
}
