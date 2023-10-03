//Contient toutes les variables susceptibles d'être changées


//Le lien (peut être symbolique) vers les graphiques
var lien=".";
var fileSep="/";

// Tableau identifiant l'xpid en fonction du nom de la chaine
var tabChaine=[];
tabChaine.push({"value":"oper","desc":"OPER","def":true});
tabChaine.push({"value":"OPER@vernaym","desc":"DEV","def":false});

//Tableau contenant les domaines
var tabDomaine=[];
tabDomaine.push({"value":"Alpes","desc":"Alpes","def":true});
tabDomaine.push({"value":"Pyrenees","desc":"Pyrénées","def":false});
tabDomaine.push({"value":"Corse","desc":"Corse","def":false});
tabDomaine.push({"value":"Massif-central","desc":"Massif-Central","def":false});
tabDomaine.push({"value":"Vosges","desc":"Vosges","def":false});
tabDomaine.push({"value":"Jura","desc":"Jura","def":false});

//Tableau contenant les massifs
var tabMassif=new Object();

//Tableau contenant les massifs des Alpes
tabMassif["alpes"]=[];
tabMassif["alpes"].push({"value":"1","desc":"Chablais","def":true});
tabMassif["alpes"].push({"value":"2","desc":"Aravis","def":false});
tabMassif["alpes"].push({"value":"3","desc":"Mont-Blanc","def":false});
tabMassif["alpes"].push({"value":"4","desc":"Bauges","def":false});
tabMassif["alpes"].push({"value":"5","desc":"Beaufortin","def":false});
tabMassif["alpes"].push({"value":"6","desc":"Haute-Tarentaise","def":false});
tabMassif["alpes"].push({"value":"7","desc":"Chartreuse","def":false});
tabMassif["alpes"].push({"value":"8","desc":"Belledonne","def":false});
tabMassif["alpes"].push({"value":"9","desc":"Maurienne","def":false});
tabMassif["alpes"].push({"value":"10","desc":"Vanoise","def":false});
tabMassif["alpes"].push({"value":"11","desc":"Haute-Maurienne","def":false});
tabMassif["alpes"].push({"value":"12","desc":"Grandes-Rousses","def":false});
tabMassif["alpes"].push({"value":"13","desc":"Thabor","def":false});
tabMassif["alpes"].push({"value":"14","desc":"Vercors","def":false});
tabMassif["alpes"].push({"value":"15","desc":"Oisans","def":false});
tabMassif["alpes"].push({"value":"16","desc":"Pelvoux","def":false});
tabMassif["alpes"].push({"value":"17","desc":"Queyras","def":false});
tabMassif["alpes"].push({"value":"18","desc":"Devoluy","def":false});
tabMassif["alpes"].push({"value":"19","desc":"Champsaur","def":false});
tabMassif["alpes"].push({"value":"20","desc":"Parpaillon","def":false});
tabMassif["alpes"].push({"value":"21","desc":"Ubaye","def":false});
tabMassif["alpes"].push({"value":"22","desc":"Verdon","def":false});
tabMassif["alpes"].push({"value":"23","desc":"Mercantour","def":false});
tabMassif["alpes"].push({"value":"24","desc":"Baronnies","def":false});
tabMassif["alpes"].push({"value":"25","desc":"Ventoux","def":false});
tabMassif["alpes"].push({"value":"26","desc":"Chiran","def":false});
tabMassif["alpes"].push({"value":"27","desc":"Alpes-azuréennes","def":false});

//Tableau contenant les massifs des Pyrénées
tabMassif["pyrenees"]=[];
tabMassif["pyrenees"].push({"value":"64","desc":"Pays Basque","def":true});
tabMassif["pyrenees"].push({"value":"65","desc":"Aspe-Ossau","def":false});
tabMassif["pyrenees"].push({"value":"66","desc":"Haute-Bigorre","def":false});
tabMassif["pyrenees"].push({"value":"67","desc":"Aure-Louron","def":false});
tabMassif["pyrenees"].push({"value":"68","desc":"Luchonnais","def":false});
tabMassif["pyrenees"].push({"value":"69","desc":"Couserans","def":false});
tabMassif["pyrenees"].push({"value":"70","desc":"Haute-Ariège","def":false});
tabMassif["pyrenees"].push({"value":"71","desc":"Andorre","def":false});
tabMassif["pyrenees"].push({"value":"72","desc":"Orlu-Saint-Barthélémy","def":false});
tabMassif["pyrenees"].push({"value":"73","desc":"Capcir-Puymorens","def":false});
tabMassif["pyrenees"].push({"value":"74","desc":"Cerdagne-Canigou","def":false});
tabMassif["pyrenees"].push({"value":"75","desc":"Pyrenees Audoises","def":false});
tabMassif["pyrenees"].push({"value":"80","desc":"Navarra","def":false});   
tabMassif["pyrenees"].push({"value":"81","desc":"Jacetania","def":false}); 
tabMassif["pyrenees"].push({"value":"82","desc":"Gallego","def":false});   
tabMassif["pyrenees"].push({"value":"83","desc":"Sobrarbe","def":false});  
tabMassif["pyrenees"].push({"value":"84","desc":"Esera","def":false});     
tabMassif["pyrenees"].push({"value":"85","desc":"Aran","def":false});      
tabMassif["pyrenees"].push({"value":"86","desc":"Ribagorcana","def":false}); 
tabMassif["pyrenees"].push({"value":"87","desc":"Pallaresa","def":false}); 
tabMassif["pyrenees"].push({"value":"88","desc":"Perafita","def":false});  
tabMassif["pyrenees"].push({"value":"89","desc":"Ter-Freser","def":false}); 
tabMassif["pyrenees"].push({"value":"90","desc":"Cadi-Moixero","def":false}); 
tabMassif["pyrenees"].push({"value":"91","desc":"Pre-Pireneu","def":false}); 

//Tableau contenant les massifs de Corse
tabMassif["corse"]=[];
tabMassif["corse"].push({"value":"40","desc":"Cinto-Rotondo","def":false});
tabMassif["corse"].push({"value":"41","desc":"Renoso-Incudine","def":false});

//Tableau contenant les massifs du Massif Central
tabMassif["massif-central"]=[];
tabMassif["massif-central"].push({"value":"50","desc":"Monts du Forez","def":false});
tabMassif["massif-central"].push({"value":"51","desc":"Sancy","def":false});
tabMassif["massif-central"].push({"value":"52","desc":"Monts du Cantal","def":false});
tabMassif["massif-central"].push({"value":"53","desc":"Aubrac","def":false});
tabMassif["massif-central"].push({"value":"54","desc":"Margeride","def":false});
tabMassif["massif-central"].push({"value":"49","desc":"Livradois","def":false});
tabMassif["massif-central"].push({"value":"59","desc":"Pilat","def":false});
tabMassif["massif-central"].push({"value":"48","desc":"Millevache","def":false});
tabMassif["massif-central"].push({"value":"60","desc":"Cévennes","def":false});
tabMassif["massif-central"].push({"value":"61","desc":"Grandes Causses","def":false});
tabMassif["massif-central"].push({"value":"62","desc":"Languedoc","def":false});

//Tableau contenant les massifs en Jura
tabMassif["jura"]=[];
tabMassif["jura"].push({"value":"55", "desc":"Haut-Doubs","def":false});
tabMassif["jura"].push({"value":"56", "desc":"Haut-Jura","def":false});
tabMassif["jura"].push({"value":"57", "desc":"Bugey Jura-Gressien","def":false});
tabMassif["jura"].push({"value":"58", "desc":"Salève","def":false});

//Tableau contenant les massifs des Vosges
tabMassif["vosges"]=[];
tabMassif["vosges"].push({"value":"45", "desc":"Donon-Champ du Feu","def":false});
tabMassif["vosges"].push({"value":"46", "desc":"Hautes-Vosges","def":false});
tabMassif["vosges"].push({"value":"47", "desc":"Ballon d'Alsace","def":false});

