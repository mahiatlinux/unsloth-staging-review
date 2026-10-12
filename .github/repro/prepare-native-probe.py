from pathlib import Path
import json
root=Path('studio/src-tauri')
p=root/'src/main.rs'
s=p.read_text(encoding="utf-8")
handler=s[s.index('        .on_webview_event('):s.index('        .on_window_event(')]
s=s.replace('fn main() {','fn studio_main() {',1)
s+='''
#[tauri::command]
fn record_drop_probe(name: String, value: serde_json::Value) -> Result<(), String> {
    let out = std::env::var("DROP_PROBE_OUT").map_err(|e| e.to_string())?;
    std::fs::write(std::path::Path::new(&out).join(format!("{name}.json")), value.to_string()).map_err(|e| e.to_string())
}
fn main() {
    tauri::Builder::default()
        .manage(native_intents::new_native_intake_state())
        .invoke_handler(tauri::generate_handler![record_drop_probe, native_intents::register_native_attachment_path])
'''+handler+'''
        .run(tauri::generate_context!()).expect("drop probe launch");
}
'''
p.write_text(s,encoding="utf-8")
c=root/'tauri.conf.json'
config=json.loads(c.read_text(encoding="utf-8"))
w=config['app']['windows'][0]
w.update(visible=True,center=False,x=300,y=120,width=700,height=500,title='Unsloth drop probe')
c.write_text(json.dumps(config,indent=2))
dist=Path('studio/frontend/dist')
(dist/'index.html').write_text('<!doctype html><html><body><h1>Unsloth native drop transport probe</h1><p>Drop the fixture here</p><pre id="result"></pre><script src="/drop-probe.js"></script></body></html>')
(dist/'drop-probe.js').write_text('''
(async () => {
 const {webview,window:win,core}=window.__TAURI__;
 const result={windowDrops:0,webviewDrops:0};
 await win.getCurrentWindow().onDragDropEvent(e=>{if(e.payload.type==='drop') result.windowDrops++;});
 await webview.getCurrentWebview().onDragDropEvent(async e=>{
  if(e.payload.type!=='drop')return;
  result.webviewDrops++;
  result.paths=e.payload.paths;
  try { result.registration=await core.invoke('register_native_attachment_path',{path:e.payload.paths[0]}); }
  catch(error){result.error=String(error);}
  await new Promise(r=>setTimeout(r,300));
  document.getElementById('result').textContent=JSON.stringify(result,null,2);
  await core.invoke('record_drop_probe',{name:'result',value:result});
 });
 await core.invoke('record_drop_probe',{name:'ready',value:{ready:true}});
})().catch(e=>{document.body.textContent=String(e)});
''')
