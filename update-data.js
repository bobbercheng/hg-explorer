#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const API_BASE = 'https://huggingface.co/api/models';
const PIPELINE = 'text-generation';
const MIN_DOWNLOADS = 1000;
const MONTHS_BACK = 4;
const OUTPUT_FILE = path.join(__dirname, 'data.json');
const BATCH_SIZE = 1000;
const RATE_LIMIT_DELAY = 500;

function parseLinkNext(linkHeader) {
  if (!linkHeader) return null;
  const match = linkHeader.match(/<([^>]+)>;\s*rel="next"/);
  return match ? match[1] : null;
}

async function main() {
  const cutoffDate = new Date();
  cutoffDate.setMonth(cutoffDate.getMonth() - MONTHS_BACK);
  cutoffDate.setDate(1);
  cutoffDate.setHours(0, 0, 0, 0);

  console.log(`Fetching text-generation models created after ${cutoffDate.toISOString().slice(0, 10)}`);
  console.log(`Minimum downloads: ${MIN_DOWNLOADS}`);

  const allModels = [];
  let fetched = 0;

  // Initial URL - sort by createdAt desc, stop when we reach the cutoff
  let url = `${API_BASE}?pipeline_tag=${PIPELINE}&sort=createdAt&direction=-1&limit=${BATCH_SIZE}&expand[]=safetensors&expand[]=inferenceProviderMapping&expand[]=downloads&expand[]=createdAt&expand[]=tags`;

  while (url) {
    try {
      const resp = await fetch(url);
      if (resp.status === 429) {
        console.log('\nRate limited, waiting 60s...');
        await new Promise(r => setTimeout(r, 60000));
        continue;
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();
      if (!Array.isArray(data) || data.length === 0) break;

      let hitCutoff = false;
      for (const m of data) {
        const created = new Date(m.createdAt);
        if (created < cutoffDate) { hitCutoff = true; break; }
        if ((m.downloads || 0) >= MIN_DOWNLOADS) {
          allModels.push(m);
        }
      }

      fetched += data.length;
      process.stdout.write(`\rFetched ${fetched} models, ${allModels.length} with ${MIN_DOWNLOADS}+ downloads...`);

      if (hitCutoff) break;

      // Get next page URL from Link header
      url = parseLinkNext(resp.headers.get('link'));
      // Preserve our expand params - the cursor URL may drop them
      if (url && !url.includes('expand')) {
        url += '&expand[]=safetensors&expand[]=inferenceProviderMapping&expand[]=downloads&expand[]=createdAt&expand[]=tags';
      }

      await new Promise(r => setTimeout(r, RATE_LIMIT_DELAY));
    } catch (e) {
      console.error('\nFetch error:', e.message);
      break;
    }
  }

  const output = {
    updatedAt: new Date().toISOString(),
    cutoffDate: cutoffDate.toISOString(),
    minDownloads: MIN_DOWNLOADS,
    totalModels: allModels.length,
    models: allModels,
  };

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));
  console.log(`\nSaved ${allModels.length} models to ${OUTPUT_FILE}`);
  console.log(`Data updated at: ${output.updatedAt}`);
}

main().catch(e => { console.error(e); process.exit(1); });
