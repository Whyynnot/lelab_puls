const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
class Element {
  constructor(id='') { this.id=id;this.value=({steps:'10',fps:'30','chart-joint':'shoulder_pan.pos'})[id]||'';this.textContent='';this.checked=true;this.disabled=false;this.hidden=false;this.children=[];this.attrs={}; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=nodes; }
  setAttribute(k,v) { this.attrs[k]=v; }
  getContext() { return new Proxy({}, {get:()=>()=>{}}); }
}
const el = id => {if (!elements.has(id)) elements.set(id,new Element(id));return elements.get(id);};
const state={active:false,mode:'sync',stopDeferred:null,stopError:false,ignoreStop:false,monitorError:false};
const profiles=[{name:'nebula_fp8',artifact:'/old',allow_robot:true,default_task:'old'},
                {name:'plate_5000_fp8',artifact:'/plate_5000',allow_robot:true,default_task:'new task'}];
const status = () => ({service:{running:true,artifact:'/plate_5000',steps:10},rollout:{inference_active:state.active},
                       robot_stop:state.robotStop,control:{mode:state.mode},offline_test:{state:'passed',inference_ms:400},service_log:'',robot_log:''});
const response = (data,ok=true) => ({ok,json:async()=>data});
const context=vm.createContext({document:{getElementById:el,createElement:()=>new Element(),hidden:false,addEventListener(){}},
  fetch:async(url,options={})=>{
    const path=url.replace('/api/deploy/','');
    if(path==='profiles')return response(profiles);
    if(path==='status')return response(status());
    if(path==='hardware-status')return response({current_devices:{follower:{accessible:true,path:'/dev/fake'},head:{accessible:true},wrist:{accessible:true}}});
    if(path==='monitor' && options.method==='POST')return response({});
    if(path==='monitor')return state.monitorError?response({detail:'monitor unavailable'},false):response({robot_active:state.active,source:null,phase:'offline_benchmark',stale:true,mode:'rtc',session:'offline',sequence:1,images:[]});
    if(path==='mode'){state.mode=JSON.parse(options.body).mode;return response({requested:{mode:state.mode}});}
    if(path==='robot/stop'){
      if(state.networkHang)return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(Object.assign(new Error('aborted'),{name:'AbortError'}))));
      if(state.stopDeferred)await state.stopDeferred;
      if(state.stopError)return response({detail:'backend timeout'},false);
      if(!state.ignoreStop)state.active=false;
      return response({stopped:true});
    }
    throw Error('Unexpected request '+path);
  },setInterval(){},setTimeout,clearTimeout,AbortController,console,Math,Date,Promise,Number,JSON,encodeURIComponent});
const settle = async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));};
(async()=>{
  vm.runInContext(script,context);await settle();
  assert.equal(el('profile').value,'plate_5000_fp8');
  assert.equal(el('task').value,'new task');
  assert.equal(el('stop').disabled,true);
  assert.match(el('robot-status').textContent,/从臂已停止.*模型仍已加载/);
  assert.match(el('monitor-status').textContent,/从臂已停止.*离线回放.*历史快照/);
  await el('rtc').onclick();assert.equal(el('rtc').attrs['aria-pressed'],'true');
  state.active=true;await vm.runInContext('refreshService()',context);
  let release;state.stopDeferred=new Promise(resolve=>{release=resolve;});
  const stopping=el('stop').onclick();await settle();
  assert.equal(el('stop').textContent,'正在停止…');assert.equal(el('stop').disabled,true);assert.equal(el('run').disabled,true);
  release();await stopping;state.stopDeferred=null;
  assert.equal(state.active,false);assert.equal(el('stop').disabled,true);assert.match(el('message').textContent,/推理已停止.*模型仍保持加载/);
  state.active=true;state.stopError=true;await vm.runInContext('refreshService()',context);await el('stop').onclick();
  assert.equal(el('stop').disabled,false);assert.match(el('message').textContent,/停止尚未确认.*backend timeout/);
  state.stopError=false;state.ignoreStop=true;await el('stop').onclick();
  assert.equal(el('stop').disabled,false);assert.match(el('message').textContent,/停止尚未确认.*后台仍报告/);
  state.ignoreStop=false;state.monitorError=true;await el('stop').onclick();
  assert.equal(el('stop').disabled,true);assert.match(el('message').textContent,/从臂推理已停止/);
  state.active=true;state.robotStop={phase:'kernel_blocked',error:'USB 清理未完成，禁止启动新任务'};
  await vm.runInContext('refreshService()',context);
  assert.match(el('robot-status').textContent,/USB 清理未完成/);assert.equal(el('run').disabled,true);
  state.networkHang=true;
  await assert.rejects(vm.runInContext("api('robot/stop',{},10)",context),/后台停止请求超时/);
  console.log(JSON.stringify({status:'passed',cases:['loaded model auto-selection','idle stop feedback','stale offline label','RTC button preserved','pending stop feedback','confirmed stopped and model remains loaded','backend stop error','reject false stop confirmation','monitor failure does not invalidate confirmed stop'],scope:'Actual page script in a DOM harness; mocked robot APIs, no hardware commands'}));
})().catch(error=>{console.error(error);process.exitCode=1;});
