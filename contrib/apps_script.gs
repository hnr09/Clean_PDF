/**
 * Clean PDF 학습 자료 수집기 (Google Apps Script)
 * 웹 페이지에서 사용자가 동의하고 보낸 페이지 이미지를 내 Google Drive 폴더에 저장합니다.
 * 설정 방법은 contrib/README.md 를 보세요.
 */
const FOLDER_NAME = 'CleanPDF_학습자료';
const MAX_BYTES = 12 * 1024 * 1024;

function doPost(e) {
  try {
    const body = e.postData && e.postData.contents;
    if (!body || body.length > MAX_BYTES) return out({ ok: false, error: 'too large' });
    const d = JSON.parse(body);
    if (!d.consent || !d.orig || !d.result) return out({ ok: false, error: 'bad request' });
    const root = getRoot();
    const name = Utilities.formatDate(new Date(), 'Asia/Seoul', 'yyyyMMdd_HHmmss') + '_' +
      Utilities.getUuid().slice(0, 8) + '_p' + (Number(d.page) || 0);
    const dir = root.createFolder(name);
    save(dir, 'orig.jpg', d.orig, 'image/jpeg');
    save(dir, 'result.jpg', d.result, 'image/jpeg');
    if (d.mask) save(dir, 'mask.png', d.mask, 'image/png');
    dir.createFile('meta.json', JSON.stringify({
      page: d.page, dpi: d.dpi, edited: !!d.edited, strength: d.strength, protect: d.protect,
      version: d.version, received: new Date().toISOString(),
    }), 'application/json');
    return out({ ok: true });
  } catch (err) {
    return out({ ok: false, error: String(err) });
  }
}

function getRoot() {
  const it = DriveApp.getFoldersByName(FOLDER_NAME);
  return it.hasNext() ? it.next() : DriveApp.createFolder(FOLDER_NAME);
}

function save(dir, name, b64, mime) {
  const bytes = Utilities.base64Decode(String(b64).replace(/^data:[^,]+,/, ''));
  dir.createFile(Utilities.newBlob(bytes, mime, name));
}

function out(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}
