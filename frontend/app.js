const $=id=>document.getElementById(id);
const sourceInput=$("sourceInput"),languageInput=$("languageInput"),analyzeBtn=$("analyzeBtn");
const progressCard=$("progressCard"),progressBar=$("progressBar"),progressPercent=$("progressPercent");
const progressTitle=$("progressTitle"),progressMessage=$("progressMessage");
const resultsSection=$("resultsSection"),emptySection=$("emptySection");
let jobId=null,currentResult=null,pollTimer=null;

function esc(s){return String(s??"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;")}
function inline(s){return esc(s).replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>").replace(/\*(.*?)\*/g,"<em>$1</em>")}
function md(text){
  if(!text)return "";
  let out="",list=false;
  const close=()=>{if(list){out+="</ul>";list=false}};
  for(let line of String(text).replace(/\r/g,"").split("\n")){
    line=line.trim(); if(!line){close();continue}
    if(line.startsWith("### ")){close();out+=`<h4>${inline(line.slice(4))}</h4>`}
    else if(line.startsWith("## ")){close();out+=`<h3>${inline(line.slice(3))}</h3>`}
    else if(line.startsWith("# ")){close();out+=`<h2>${inline(line.slice(2))}</h2>`}
    else if(/^[-*•]\s/.test(line)||/^\d+\.\s/.test(line)){
      if(!list){out+="<ul>";list=true}
      out+=`<li>${inline(line.replace(/^[-*•]\s/,"").replace(/^\d+\.\s/,""))}</li>`
    }else{close();out+=`<p>${inline(line)}</p>`}
  }
  close();return out
}

function setProgress(percent,title,message,stage){
  progressBar.style.width=percent+"%";
  progressPercent.textContent=percent+"%";
  progressTitle.textContent=title;
  progressMessage.textContent=message;
  const stages=["download","transcribe","analyze","rag"];
  stages.forEach((s,i)=>{
    const el=$("stage-"+s);
    el.classList.toggle("done",i<stage);
    el.classList.toggle("active",i===stage);
  });
}

async function startAnalysis(){
  const source=sourceInput.value.trim();
  if(!source){alert("Please enter a YouTube URL or local file path.");return}
  resultsSection.classList.add("hidden");emptySection.classList.add("hidden");
  progressCard.classList.remove("hidden");analyzeBtn.disabled=true;
  setProgress(3,"Starting analysis...","Preparing the AI pipeline.",0);
  try{
    const r=await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({source,language:languageInput.value})});
    const d=await r.json();if(!r.ok)throw Error(d.detail||"Unable to start analysis.");
    jobId=d.job_id;pollJob();
  }catch(e){fail(e.message)}
}

async function pollJob(){
  try{
    const r=await fetch(`/api/status/${jobId}`),d=await r.json();
    if(!r.ok)throw Error(d.detail||"Status check failed.");
    const p=Number(d.progress??d.percent??0);
    const status=String(d.status||"processing").toLowerCase();
    const msg=d.message||"Processing...";
    let stage=0;
    if(p>=70||/summary|extract|analys/i.test(msg))stage=2;
    else if(p>=20||/transcrib|whisper|sarvam/i.test(msg))stage=1;
    if(p>=90||/rag|vector|embed|complete/i.test(msg))stage=3;
    if(status==="queued")setProgress(Math.max(2,p),"Waiting in queue...",msg,0);
    else setProgress(Math.max(5,p),status==="completed"?"Analysis complete":msg,msg,stage);
    if(status==="completed"){setProgress(100,"Analysis complete","Meeting intelligence is ready.",4);return loadResult()}
    if(status==="failed")throw Error(msg||"Analysis failed.");
    pollTimer=setTimeout(pollJob,1200);
  }catch(e){fail(e.message)}
}

async function loadResult(){
  const r=await fetch(`/api/result/${jobId}`),d=await r.json();
  if(!r.ok)throw Error(d.detail||"Unable to load result.");
  currentResult=d;
  $("meetingTitle").textContent=d.title||"Meeting Analysis";
  $("summaryContent").innerHTML=md(d.summary||"No summary available.");
  $("actionItemsContent").innerHTML=md(d.action_items||"No action items found.");
  $("decisionsContent").innerHTML=md(d.key_decisions||"No key decisions found.");
  $("questionsContent").innerHTML=md(d.open_questions||"No open questions found.");
  $("transcriptContent").textContent=d.transcript||"Transcript unavailable.";
  $("chatContainer").innerHTML='<div class="chat-empty">✦ &nbsp; Ask anything about your meeting transcript.</div>';
  setTimeout(()=>{progressCard.classList.add("hidden");resultsSection.classList.remove("hidden");analyzeBtn.disabled=false},500);
}

function fail(message){
  clearTimeout(pollTimer);progressCard.classList.add("hidden");emptySection.classList.remove("hidden");
  analyzeBtn.disabled=false;alert("Error: "+message);
}

analyzeBtn.addEventListener("click",startAnalysis);
sourceInput.addEventListener("keydown",e=>{if(e.key==="Enter")startAnalysis()});
$("copySummaryBtn").addEventListener("click",async()=>{await navigator.clipboard.writeText(currentResult?.summary||"");$("copySummaryBtn").textContent="Copied ✓";setTimeout(()=>$("copySummaryBtn").textContent="Copy summary",1500)});
$("transcriptButton").addEventListener("click",()=>{$("transcriptSection").classList.toggle("hidden")});

$("askButton").addEventListener("click",ask);
$("chatInput").addEventListener("keydown",e=>{if(e.key==="Enter")ask()});
async function ask(){
  const q=$("chatInput").value.trim();if(!q)return;
  if(!jobId){addChat("assistant","Please analyze a meeting first.");return}
  addChat("user",q);$("chatInput").value="";$("askButton").disabled=true;$("askButton").textContent="Thinking...";
  try{
    const r=await fetch(`/api/chat/${jobId}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
    const d=await r.json();if(!r.ok)throw Error(d.detail||"RAG request failed.");addChat("assistant",d.answer);
  }catch(e){addChat("assistant","I couldn't answer that question. Please check the backend terminal for the RAG error.")}
  finally{$("askButton").disabled=false;$("askButton").textContent="Ask →"}
}
function addChat(role,text){
  const c=$("chatContainer"),empty=c.querySelector(".chat-empty");if(empty)empty.remove();
  const el=document.createElement("div");el.className="chat-message "+role;
  el.innerHTML=`<div class="chat-role">${role==="user"?"YOU":"AI ASSISTANT"}</div>${role==="assistant"?md(text):`<p>${esc(text)}</p>`}`;
  c.appendChild(el);c.scrollTop=c.scrollHeight;
}
