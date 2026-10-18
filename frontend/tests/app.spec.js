// Intercepted responses exist only in tests. They are not model evaluation results.
import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { mkdir } from 'node:fs/promises';

const fixture = fileURLToPath(new URL('./fixtures/design-leaf.png', import.meta.url));
const screenshots = fileURLToPath(new URL('../../docs/screenshots/', import.meta.url));
const prediction = {
  predicted_class: 'Early Blight', confidence: 0.7,
  probabilities: { Healthy: 0.08, 'Early Blight': 0.7, 'Late Blight': 0.12, 'Leaf Mold': 0.1 },
  model_identifier: 'UI-TEST-FIXTURE-NOT-REAL-INFERENCE',
};

async function upload(page) {
  await page.getByLabel('Upload leaf image', {exact: true}).setInputFiles(fixture);
  await expect(page.getByRole('heading', {name: 'Preview Image'})).toBeVisible();
}
async function noOverflow(page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
}

for (const width of [375, 390, 430]) {
  test(`four screens and navigation at ${width}px`, async ({ page }) => {
    await mkdir(screenshots, {recursive: true});
    await page.setViewportSize({width, height:844});
    let respond;
    const gate = new Promise(resolve => { respond = resolve; });
    await page.route('**/predict', async route => {
      await gate;
      await route.fulfill({json: prediction});
    });
    await page.goto('/');
    await expect(page.getByRole('heading', {name: 'Tomato Leaf Disease Recognition'})).toBeVisible();
    await noOverflow(page);
    if (process.env.UPDATE_SCREENSHOTS === '1') await page.screenshot({path:`${screenshots}/${width}-01-upload.png`, fullPage:true});
    await expect(page.getByLabel('Take leaf photo')).toHaveAttribute('capture','environment');
    await upload(page);
    await expect(page.getByAltText('Your selected tomato leaf')).toHaveAttribute('src', /^blob:/);
    await noOverflow(page);
    if (process.env.UPDATE_SCREENSHOTS === '1') await page.screenshot({path:`${screenshots}/${width}-02-preview.png`, fullPage:true});
    await page.getByRole('button', {name:'Analyze Image', exact:true}).click();
    await expect(page.getByRole('heading', {name:'Analyzing Image', exact:true})).toBeVisible();
    await expect(page.getByRole('status')).toContainText('Waiting');
    await expect(page.getByText('70%',{exact:true})).toHaveCount(0);
    await noOverflow(page);
    if (process.env.UPDATE_SCREENSHOTS === '1') await page.screenshot({path:`${screenshots}/${width}-03-analyzing.png`, fullPage:true});
    respond();
    await expect(page.getByRole('heading', {name:'Result',exact:true})).toBeVisible();
    await expect(page.getByRole('meter')).toHaveCount(4);
    await expect(page.getByText('Confidence: 70.0%', {exact:true})).toBeVisible();
    await expect(page.getByText('Septoria Leaf Spot',{exact:true})).toHaveCount(0);
    await noOverflow(page);
    if (process.env.UPDATE_SCREENSHOTS === '1') await page.screenshot({path:`${screenshots}/${width}-04-result.png`, fullPage:true});
    await page.getByRole('button', {name:'View Disease Information'}).click();
    await expect(page.getByRole('dialog')).toContainText('concentric rings');
    await page.getByRole('button', {name:'Close',exact:true}).click();
    await page.getByRole('button', {name:'Analyze Another Image'}).click();
    await expect(page.getByRole('button',{name:'Upload Image',exact:true})).toBeVisible();
  });
}

test('server error, retry, back and image replacement', async ({ page }) => {
  await page.setViewportSize({width:390,height:844});
  let attempts = 0;
  await page.route('**/predict', route => {
    attempts++;
    return attempts === 1 ? route.fulfill({status:503,json:{detail:'Model checkpoint unavailable'}}) : route.fulfill({json:prediction});
  });
  await page.goto('/');
  await upload(page);
  await page.getByRole('button',{name:'Analyze Image',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('Model checkpoint unavailable');
  await page.getByRole('button',{name:'Try Again'}).click();
  await expect(page.getByRole('heading',{name:'Result',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Back',exact:true}).click();
  await page.getByRole('button',{name:'Choose Another Image'}).click();
  await page.getByLabel('Upload leaf image',{exact:true}).setInputFiles(fixture);
  await expect(page.getByRole('heading',{name:'Preview Image'})).toBeVisible();
  await page.getByRole('button',{name:'Back',exact:true}).click();
  await expect(page.getByRole('button',{name:'Upload Image',exact:true})).toBeVisible();
});

test('reject unsupported, empty, oversized files and invalid API results', async ({ page }) => {
  await page.goto('/');
  const input=page.getByLabel('Upload leaf image',{exact:true});
  await input.setInputFiles({name:'leaf.txt',mimeType:'text/plain',buffer:Buffer.from('bad')});
  await expect(page.getByRole('alert')).toContainText('JPG or PNG');
  await input.setInputFiles({name:'empty.png',mimeType:'image/png',buffer:Buffer.alloc(0)});
  await expect(page.getByRole('alert')).toContainText('empty');
  await input.setInputFiles({name:'large.png',mimeType:'image/png',buffer:Buffer.alloc(10*1024*1024+1)});
  await expect(page.getByRole('alert')).toContainText('exceeds 10 MB');
  await page.route('**/predict',route=>route.fulfill({json:{...prediction,probabilities:{Healthy:1}}}));
  await upload(page);
  await page.getByRole('button',{name:'Analyze Image',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('invalid prediction');
});

test('camera input accepts an image and about dialog explains offline limits', async ({page})=>{
  await page.goto('/');
  await page.getByRole('button',{name:'About TomatoCare'}).click();
  await expect(page.getByRole('dialog')).toContainText('does not enable offline inference');
  await page.keyboard.press('Escape');
  await page.getByLabel('Take leaf photo',{exact:true}).setInputFiles(fixture);
  await expect(page.getByRole('heading',{name:'Preview Image'})).toBeVisible();
});

test('offline AI request shows a clear error without fake results', async ({page,context})=>{
  await page.goto('/');
  await upload(page);
  await context.setOffline(true);
  await page.getByRole('button',{name:'Analyze Image',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('Internet connection is required');
  await expect(page.getByRole('heading',{name:'Result',exact:true})).toHaveCount(0);
  await context.setOffline(false);
  await page.getByRole('button',{name:'Back to Preview'}).click();
  await expect(page.getByRole('heading',{name:'Preview Image'})).toBeVisible();
});


test('cold-start HTML error is clear and a later retry succeeds', async ({page})=>{
  let attempts=0;
  await page.route('**/predict', route=>++attempts===1
    ? route.fulfill({status:503,contentType:'text/html',body:'Service starting'})
    : route.fulfill({json:prediction}));
  await page.goto('/');
  await upload(page);
  await page.getByRole('button',{name:'Analyze Image',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('may be starting');
  await page.getByRole('button',{name:'Try Again'}).click();
  await expect(page.getByRole('heading',{name:'Result',exact:true})).toBeVisible();
});

test('request timeout shows retry without changing the four-screen flow', async ({page})=>{
  await page.clock.install();
  await page.route('**/predict',()=>{});
  await page.goto('/');
  await upload(page);
  await page.getByRole('button',{name:'Analyze Image',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Analyzing Image',exact:true})).toBeVisible();
  await page.clock.fastForward(120001);
  await expect(page.getByRole('alert')).toContainText('timed out');
  await expect(page.getByRole('button',{name:'Try Again'})).toBeVisible();
});
