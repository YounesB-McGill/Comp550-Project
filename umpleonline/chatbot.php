<?php header('X-Frame-Options: ALLOWALL'); ?>

<!DOCTYPE html>
<html>
    
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title>Test webpage for chatbot</title>
        <meta name="description" content="">
        <style>
            .bubble {
                padding: 10px;
                margin: 10px 0;
                font-family: Arial, sans-serif
            }
            
            .usermsg {
                border: 2px solid #00ff00;
                background-color: #ccffcc;
            }
            
            .chatbot {
                border: 2px solid #004d99;
                background-color: #ccf5ff;
            }
        </style>

        <script type="text/javascript">
            
            function addMessage() {
              var userInput = document.getElementById('userinput').value;
              if (userInput == '') return;
              
              document.getElementById('chathistory').innerHTML += 
                  ('<div class="bubble usermsg">' + userInput + '</div>');
              document.getElementById('userinput').value = ""
              
              // TODO Actually call chatbot
              wait(400);
              chatbotReply();

              window.scrollTo(0, document.body.scrollHeight);
            }

            function chatbotReply() {
                document.getElementById('chathistory').innerHTML += 
                    ('<div class="bubble chatbot"><b>Chatbot:</b> I will reply once I am implemented :)</div>');
                window.scrollTo(0, document.body.scrollHeight);
            }

            function callChatbot(userInput) {
              var xhttp = new XMLHttpRequest();
              var reply = "No reply received."
              xhttp.onreadystatechange = function() {
                if (this.readyState == 4 && this.status == 200) {
                    document.getElementById('chathistory').innerHTML += ('<div> <h2>' + 
                            reply + '</h2> </div>');
                }
              };
              xhttp.open("GET", "ajax_info.txt", true);
              xhttp.send();
            }

            function wait(ms){
               var start = new Date().getTime();
               var end = start;
               while(end < start + ms) {
                 end = new Date().getTime();
              }
            }

            function debug4() {
              window.parent.Action.directAddClass("Pizza");
            }
        </script>
        
    </head>
    <body>
        <h1 style="font-family: sans-serif;">Chatbot</h1>
        
        <input type="button" onclick="debug4()" value="debug" id="debug"></input>
        
        <div id="chathistory"></div>

        <div>
            <input type="text" id="userinput"></input>
            <input 
              type="button" 
              onclick='addMessage()' 
              value="Send" 
              id="sendbutton"
            />
            <!-- Hit Enter to Send -->
            <script type="text/javascript">
              var input = document.getElementById("userinput");
              input.addEventListener("keyup", function(event) {
                event.preventDefault();
                if (event.keyCode === 13) { // ENTER
                  document.getElementById("sendbutton").click();
                }
              });
            </script>
        </div>
        
    </body>
</html>





