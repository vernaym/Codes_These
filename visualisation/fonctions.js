//Contient le code permettant de remplir les menus, de manipuler les dates, de choisir le nom de l'image et de la charger
// 27/10/2014

//Tableau contenant les mois
tabMois=new Object();
tabMois["01"]="janvier";
tabMois["02"]="février";
tabMois["03"]="mars";
tabMois["04"]="avril";
tabMois["05"]="mai";
tabMois["06"]="juin";
tabMois["07"]="juillet";
tabMois["08"]="août";
tabMois["09"]="septembre";
tabMois["10"]="octobre";
tabMois["11"]="novembre";
tabMois["12"]="décembre";

//Dictionnaire des domaines
tabDomaines=new Object();
tabDomaines["Alpes"]="alp";
//tabDomaines["Pyrenees"]="pyr";
//tabDomaines["Corse"]="cor";
//tabDomaines["Massif-central"]="mac";
//tabDomaines["Vosges"]="vog";
//tabDomaines["Jura"]="jur";


//Variable pour la gestion des exceptions
var nbExcept=1;

//Fonction qui est appelée lors du chargement de la page
function chargement(){

  //On met en place les événements sur les différents menus
  //document.getElementById("selectDomaine").addEventListener("change",function(){remplissageSelectMassif()});
  //document.getElementById("selectDate").addEventListener("change",function(){chargeImage()});
  document.getElementsByName("valid")[0].addEventListener("click",function(){chargeImage();});
//TODO
//  document.getElementsByName("avance")[0].addEventListener("click",function(){avanceDate();remplissageSelectJournee();chargeImage()});
//  document.getElementsByName("recule")[0].addEventListener("click",function(){reculeDate();remplissageSelectJournee();chargeImage()});
  
  //Gestion des erreurs si l'image est manquante
//  document.getElementById("img").addEventListener("error",function(){imageManquante();});
  //On remplit les select
  
  remplissageSelect();
  //remplissageSelectDate();

}  // Fin chargement

//Fonction permettant de remplir le menu des massifs
function remplissageSelectMassif(){
  
  
  //On récupère le massif qui est sélectionné
  var domaine=document.getElementById("selectDomaine").value;
  //var tabMassifTmp=tabMassif[domaine];
} //Fin remplissageSelectMassif


//Fonction permettant de remplir un select à partir d'un tableau de données associatif simple
//Permet de remplie les menus de variables, de scores, de domaines et d'altitudes
//L'identifiant du select à remplir ainsi que le tableau de données correspondant est en argument
function remplissageSelectSimple(idSelect,tabData){

  //On récupère le select
  sel=document.getElementById(idSelect);

  //On récupère la valeur du select
  valueBefore=sel.value;

  //On efface ce qu'il y a dans le select
  sel.innerHTML="";

  //On parcourt le select en le remplissant au fur et à mesure
  for (var i=0;i<tabData.length;i++){
    value=tabData[i]["value"];
    desc=tabData[i]["desc"];
    selected=tabData[i]["def"]
    opt = document.createElement("option");
    opt.value=value;
    opt.text=desc;
    opt.selected=selected;
    sel.appendChild(opt);
  }

  //On reselectionne la valeur précédemment sélectionné
  choixSelect(idSelect,valueBefore);  

} //Fin remplissageSelectSimple

//Permet de remplir le menu des dates
function remplissageSelectDate(){

  //On récupère la période
  //var periode=document.getElementById("selectPeriode").value;

  //On récupère le type de graphique
  //var typeGraphe=document.getElementById("selectGraphe").value;

  //On la table qui remplira le select
  tabDate=[];

  //On fixe la date
  var today=new Date();

  today.setDate(today.getDate());
  //On parcourt les 365 jours précédents
  for (var i=0;i<365;i++){
    def=false;
    if(i==0){
	def=true;
	}
    var year=(today.getFullYear()).toString();
    var month=addZero(today.getMonth()+1,2);
    var day=addZero(today.getDate(),2);
    var value=year+month+day
    var desc=day+"/"+month+"/"+year;
    tabDate.push({"value":value,"desc":desc,"def":def});
    today.setDate(today.getDate()-1);
  }  

  //On remplit le select de date
  remplissageSelectSimple("selectDate",tabDate);

}

//Permet de remplir le menu du choix de la date analysée
//function remplissageSelectJournee(){
//
//  //On récupère la date d'analyse
//  tabDate=[];
//  var ana=document.getElementById("selectDate").value;
//  var year        = ana.substring(0,4);
//  var month       = ana.substring(4,6)-1;
//  var day         = ana.substring(6,8);
//  var dateana     = new Date(year, month, day);
////  var resana=document.getElementById("selectReseau").value;
//  var hour = 6;
//  var endyear=(dateana.getFullYear()).toString();
//  var endmonth=addZero(dateana.getMonth()+1,2);
//  var endday=addZero(dateana.getDate(),2);
//  var value=endyear+endmonth+endday;
//  dateana.setDate(dateana.getDate()-1);
//  var year=(dateana.getFullYear()).toString();
//  var month=addZero(dateana.getMonth()+1,2);
//  var day=addZero(dateana.getDate(),2);
//  var desc=day+"/"+month+"/"+year+" 6h to "+endday+"/"+endmonth+"/"+endyear+" 6h";
//  tabDate.push({"value":value,"desc":desc,"def":def});
//  }  
//
//  remplissageSelectSimple("selectJournee",tabDate);
//  
//}

//Fonction permettant de reculer la date
function reculDate(){
date=document.getElementById("selectJournee").value
//TODO : reculer la date de 1 journée
choixSelect("selectJournee",newdate)
}

//Fonction permettant d'avancer le réseau
function avanceDate(){
date=document.getElementById("selectJournee").value
//TODO : avancer la date de 1 journée
choixSelect("selectJournee",newdate)
}

//Fonction qui est appelée pour mettre à jour l'ensemble des select (au chargement ou au changement de type de graphique)
function remplissageSelect(){

  //remplissageSelectSimple("selectDomaine",tabDomaine); 
  //Remplissage des dates d'analyse
  remplissageSelectDate();
}

//Fonction permettant de construire le nom de l'image et de la charger
function chargeImage(){

  //On commence par récupérer les valeurs de tous les select
  //var domaine=document.getElementById("selectDomaine").value;
  var date=document.getElementById("selectDate").value;

  //On construit le nom de l'image
  var nomMapCourt;
  var nomTableCourt;

  //nomMapCourt="map_"+massif+"_"+date;
  //nomTableCourt="table_"+massif+"_"+date;

  //On ajoute les extensions au nom du graphique
  //var nomMap=lien+fileSep+tabDomaines[domaine]+fileSep+'precipitation_'+date+".html";
  var nomMap=lien+fileSep+'figures'+fileSep+'precipitation_'+date+".html";
  //var nomTable=lien+fileSep+xpid+fileSep+tabDomaines[domaine]+fileSep+daterun+reseau+fileSep+nomTableCourt+".png";
  //var nomMap="/cnrm/mrns/users/NO_SAVE/vernaym/ANTILOPE/test.html";

  //On fixe l'image
  var map=document.getElementById("map");
  nbExcept=1;
  console.log(nomMap);
  //map.src=nomMap;
  //map.title=nomMapCourt;
  map.data=nomMap;
  map.width="1800";
  map.height="900";

  //On affiche l'image
  document.getElementById("zoneImage").style.display="inline";
}

//Le code qui s'execute si l'image n'a pas été trouvées
//function imageManquante(){

  //On affiche une autre image
//  var image=document.getElementById("img");
//  if (nbExcept>0.5){  //Pour éviter le chaînage des erreurs
//    image.src="img_erreur.png";
//  }
//  nbExcept+=-1;

//}

//Fonction permettant d'ajouter des zéros à un nombre (padding)
function addZero(x,n) {
  if (x.toString().length < n) {
    x = "0" + x;
  }
  return x;
}

//Fonction renvoyant une chaîne "aléatoire" basée sur le temps en millisecond
function getAlea(){
  var d = new Date()
  var h = addZero(d.getHours(), 2);
  var m = addZero(d.getMinutes(), 2);
  var s = addZero(d.getSeconds(), 2);
  var ms = addZero(d.getMilliseconds(), 3);
  return h+m+s+ms;
}


//Fonction permettant de sélectionner une valeur d'un select
//Les arguments sont l'identifiant du select et la valeur à sélectionner (si disponible)
function choixSelect(idSelect,value){

  //On récupère le select
  sel=document.getElementById(idSelect);

  //On parcourt le select pour voir si la valeur existe, si c'est le cas, on la sélectionne
  for(var i=0;i<sel.options.length;i++){
    var opt=sel.options[i];
    if(opt.value==value){
      opt.selected=true;
    }
  }

}  //Fin choixSelect



