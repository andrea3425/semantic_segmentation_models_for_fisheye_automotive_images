from IPython.display import display, HTML, Javascript
import time

# Aggiungere un elemento per il cronometro e per il contatore del set
html_str = """
<style>
    .customFont {
        font-family: monospace;
        font-size: 14px; /* Dimensione standard in Colab */
    }
</style>
<div id="timer" class="customFont" style="text-align: left; margin-bottom: 10px;">00:00:00</div>
<div id="set" class="customFont" style="text-align: left; margin-bottom: 10px;">Set 0: 0/0</div>
<div style="width: 100%; display: flex; align-items: center;">
  <div id="myProgress" style="width: 95%; background-color: grey;">
    <div id="myBar" style="width: 1%; height: 30px; background-color: #4bcc99;"></div>
  </div>
  <div id="progressPercent" class="customFont" style="width: 5%; text-align: right;">0%</div>
</div>
"""

# Aggiornare la funzione JavaScript per accettare un testo personalizzato per il set
js_str = """
function updateProgress(value, seconds, currentSet, totalSet, setText) {
    var elem = document.getElementById("myBar");
    var textElem = document.getElementById("progressPercent");
    var timerElem = document.getElementById("timer");
    var setElem = document.getElementById("set");
    
    elem.style.width = value + '%';
    textElem.innerHTML = value + '%';
    
    // Calcolare ore, minuti e secondi
    var hours = Math.floor(seconds / 3600);
    var minutes = Math.floor((seconds % 3600) / 60);
    var seconds = seconds % 60;
    
    // Formattare il tempo per il cronometro
    var formattedTime = 
        (hours < 10 ? "0" + hours : hours) + ":" + 
        (minutes < 10 ? "0" + minutes : minutes) + ":" + 
        (seconds < 10 ? "0" + seconds : seconds);
    
    timerElem.innerHTML = formattedTime;
    
    // Aggiornare il contatore del set con il testo personalizzato
    setElem.innerHTML = setText + ": " + currentSet + "/" + totalSet;
}
"""

# Funzione per eseguire lo script JavaScript dall'interno di Python con i nuovi parametri
def move_progress(value, elapsed_time, current, total, set_text):
    display(Javascript('updateProgress({}, {}, {}, {}, "{}")'.format(value, elapsed_time, current, total, set_text)))

