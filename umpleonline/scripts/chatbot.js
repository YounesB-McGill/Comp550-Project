var jsonResponse;
var userResponse = '';
var intent;
var modelLog = '';
var modelClassJsons = {};
var umpleCode;
var Action = window.parent.Action;
var Page = window.parent.Page;
var Layout = window.parent.Layout;
var DiagramEdit = window.parent.DiagramEdit;

const defaultClassPositions = [[ 50,  50], [250,  50], [450,  50],
                               [ 50, 200], [250, 200], [450, 200],
                               [ 50, 350], [250, 350], [450, 350],
                               [ 50, 500], [250, 500], [450, 500]];

const OLD_SERVER_URL = "http://localhost:8002/watson/app.js"; // Remove this later
const SERVER_URL = "http://localhost:8003/process-input";

const TIMEOUT_BETWEEN_ACTIONS = 500; // milliseconds

const defaultUserResponse = "Sorry, I didn't get that, can you rephrase? :)";
var numClasses = 0;

var classesAdded = [];

Layout.setUmpleCanvasSize(1200, 900);

function addMessage() {
  var userInput = document.getElementById('userinput').value;
  if (userInput == '') return;

  document.getElementById('chathistory').innerHTML +=
    ('<div class="bubble usermsg">' + userInput + '</div>');
  document.getElementById('userinput').value = "";

  callChatbot(userInput);

  window.scrollTo(0, document.body.scrollHeight);
}

function chatbotReply(response) {
  if (response) {
    document.getElementById('chathistory').innerHTML +=
      ('<div class="bubble chatbot"><b>Chatbot:</b> ' + response + '</div>');
  } else {
    document.getElementById('chathistory').innerHTML +=
      ('<div class="bubble chatbot"><b>Chatbot:</b>' + defaultUserResponse + '</div>');
  }
  window.scrollTo(0, document.body.scrollHeight);
}

function updateModelLog() {
  var list = window.parent.document.getElementsByTagName('h');
  for (var i = 0; i < list.length; i++){
    var postCmd = list[i].firstChild.nodeValue;
    modelLog += postCmd + "\n\n";
    
    if (postCmd.includes("action=addClass")) {
      postCmd = postCmd.replace("action=addClass&actionCode=", "");
      var classInfo = JSON.parse(postCmd);
      modelClassJsons[classInfo.name] = classInfo;
    }
  }
}

function updateUmpleCode() {
  umpleCode = Page.getUmpleCode();
  console.log(umpleCode);
}

function chatbotAction() {
  console.log(`jsonResponse: ${typeof(jsonResponse)} = ${JSON.stringify(jsonResponse)}`);
  intent = (((jsonResponse||{}).intents||[])[0]||{}).intent || 'unknown';
  console.log("intent: " + intent);
  
  const action = {
    'create_class': addClass,
    'add_attribute': addAttribute,
    'create_inheritance': addInheritance,
    'create_association': addAssociation,
    'create_composition': addComposition,
    'return_error_to_user': returnErrorToUser,
    'unknown': () => {}
  };

  action[intent]();
  updateModelLog();
}

function callChatbot(userInput) {
  $(document).ready(function() {
    var jqueryXHR = $.ajax({
      type: 'POST',
      url: SERVER_URL,
      dataType: 'json',
      data: {
        'param': '0',
        'input': userInput
      },
    });
    console.log(userInput);
    console.log(jsonResponse);
    jqueryXHR.always(function(resp) {
      jsonResponse = resp;
      console.log(resp);
      chatbotAction();
      chatbotReply(userResponse);
    });
  });
  setTimeout(updateUmpleCode, 2000);
}

function wait(ms) {
  var start = new Date().getTime();
  var end = start;
  while (end < start + ms) {
    end = new Date().getTime();
  }
}

function addClass() {
  userResponse = jsonResponse.output.text[0];
  const className = jsonResponse.entities[0].value || 'NewClass';
  addClassWithName(className);
}

function addClassWithName(className) {
  if (!Page.getUmpleCode().includes(`class ${className}`)) {
    Action.directAddClass(firstLetterUppercase(className), defaultClassPositions[numClasses]);
    updateModelLog();
  } else {
    userResponse = `${className} is already created, so let's not make it again.`;  
  }

  if (!classesAdded.includes(className)) {
    classesAdded.push(className);
    numClasses++;
  }
}

function addAttribute() {
  userResponse = jsonResponse.output[0].text;
  const className = firstLetterUppercase(jsonResponse.entities[0].value) || 'NewClass';

  var timeout = 0;
  if (!classesAdded.includes(className)) {
    addClassWithName(className);
    timeout = TIMEOUT_BETWEEN_ACTIONS;
  } 

  const attributeName = jsonResponse.entities[1].value || 'attributeName';
  var json = modelClassJsons[className];
  var attributes = {
    type: "String", name: attributeName, textColor: "black", aColor: "black", newType: "String", newName: attributeName
  };
  if (typeof json.attributes === "undefined") { 
    json.attributes = [attributes];
  } else {
    json.attributes.push(attributes);
  }
  var request = "action=editClass&actionCode=" + JSON.stringify(json);
  setTimeout(() => Action.ajax(Action.directUpdateCommandCallback, request), timeout);
}

function addInheritance() {
  userResponse = jsonResponse.output.text[0];
  fixInheritanceUserResponse();
  const childClassName = firstLetterUppercase(jsonResponse.entities[0].value) || 'SubClass';
  const parentClassName = firstLetterUppercase(jsonResponse.entities[1].value) || 'SuperClass';

  var timeout = 0;
  if (!classesAdded.includes(parentClassName)) {
    addClassWithName(parentClassName);
    timeout += TIMEOUT_BETWEEN_ACTIONS;
  }
  if (!classesAdded.includes(childClassName)) {
    setTimeout(() => addClassWithName(childClassName), timeout);
    timeout += TIMEOUT_BETWEEN_ACTIONS;
  }

  console.log('par:'+parentClassName);
  var request = `action=addGeneralization&actionCode={"parentId": ${parentClassName}, "childId": ${childClassName}}`;
  setTimeout(() => Action.ajax(Action.directUpdateCommandCallback, request), timeout);
}

function addAssociation() {
  userResponse = jsonResponse.output[0].text;
  const className1 = firstLetterUppercase(jsonResponse.entities[0].value) || 'NewClass1';
  const className2 = firstLetterUppercase(jsonResponse.entities[1].value) || 'NewClass2';

  var timeout = 0;
  if (!classesAdded.includes(className1)) {
    addClassWithName(className1);
    timeout += TIMEOUT_BETWEEN_ACTIONS;
  }
  if (!classesAdded.includes(className2)) {
    setTimeout(() => addClassWithName(className2), timeout);
    timeout += TIMEOUT_BETWEEN_ACTIONS;
  }

  setTimeout(() => Action.ajax(Action.directUpdateCommandCallback,
    `action=addAssociation&actionCode={  
      "classOnePosition": ${JSON.stringify(modelClassJsons[className1].position)},
      "classTwoPosition": ${JSON.stringify(modelClassJsons[className2].position)},
      "offsetOnePosition":{  
        "x":"50",
        "y":"50",
        "width":"0",
        "height":"0"
      },
      "offsetTwoPosition":{  
        "x":"50",
        "y":"50",
        "width":"0",
        "height":"0"
      },
      "multiplicityOne":"*",
      "multiplicityTwo":"*",
      "name":"${className1}__${className2}",
      "roleOne":"",
      "roleTwo":"",
      "isSymmetricReflexive":"false",
      "isLeftNavigable":"true",
      "isRightNavigable":"true",
      "isLeftComposition":"false",
      "isRightComposition":"false",
      "color":"black",
      "id":"${"umpleAssociation_" + 2*numClasses}",
      "classOneId":"${className1}",
      "classTwoId":"${className2}"
    }`
  ), timeout);
}

function addComposition() {
  userResponse = jsonResponse.output.text[0];
  const wholeClassName = firstLetterUppercase(jsonResponse.context.varContainer) || 'Whole';
  var partClassName = (jsonResponse.context.varPart).trim()  || 'Part';

  if (partClassName.substr(-1) === 's' && modelClassJsons[partClassName] === undefined) {
    partClassName = partClassName.slice(0, -1);
  }
  partClassName = firstLetterUppercase(partClassName);

  var timeout = 0;
  if (!classesAdded.includes(wholeClassName)) {
    addClassWithName(wholeClassName);
    timeout += TIMEOUT_BETWEEN_ACTIONS;
  }
  if (!classesAdded.includes(partClassName)) {
    setTimeout(() => addClassWithName(partClassName), timeout);
    timeout += TIMEOUT_BETWEEN_ACTIONS;
  }

  setTimeout(() => {
    var lines = Page.getUmpleCode().split('\n');
    umpleCode = '';
    var done = false;
    for (var i = 0; i < lines.length; i++) {
      if (!done && lines[i].includes('class ' + wholeClassName)) {
        lines.splice(i+2, 0, `  1 <@>- * ${partClassName};`);
        done = true;
      }
      umpleCode += lines[i] + '\n';
    }
    Page.setUmpleCode(umpleCode);
    Action.updateUmpleDiagram();
  }, timeout);
}

function returnErrorToUser() {
  userResponse = jsonResponse.output.text;
}

function debug4() {
  Action.directAddClass("School", defaultClassPositions[numClasses++]);
}

function firstLetterUppercase(input) {
  input = input.trim();
  return input[0].toUpperCase() + input.substring(1);
}

/**
 * Temporary workaround a bug in Watson Assistant.
 * When a sentence is cut off like this "Okay! I've created an inheritance between Fruit and,"
 * it will attempt to add the missing word.
 */
function fixInheritanceUserResponse() {
  var words = userResponse.split(' ');
  if (words[words.length - 1] === '') {
    userResponse += jsonResponse.entities[0].value + '.';
  }
}
