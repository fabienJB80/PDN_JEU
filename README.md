
**PDN_JEU**

Lecteurs PDN pour le jeu de dames international (100 cases).

 - Projet 100% HTML + JavaScript. 
 - Aucun backend.

------------------------------------------------------------------------

**1. index.html — *Lecteur autonome***

Démo :
https://fabienjb80.github.io/PDN_JEU/index.html

Utilisation directe

-   Charger un fichier .pdn
-   Coller un PDN
-   Navigation coup par coup

Intégration via iframe

    <iframe
      src="https://fabienjb80.github.io/PDN_JEU/index.html"
      width="100%"
      height="650"
      style="border:1px solid #ccc;border-radius:10px;">
    </iframe>

------------------------------------------------------------------------

**2. SimplyPDN — *Lecture d’un PDN distant***

Lecture d’un fichier PDN via paramètre d’URL ?pdn=.

    <iframe
      src="https://fabienjb80.github.io/PDN_JEU/index.html?pdn=https://codeberg.org/UTILISATEUR/DEPOT/raw/branch/main/parties/ma_partie.pdn"
      width="100%"
      height="650"
      style="border:1px solid #ccc;border-radius:10px;">
    </iframe>

Flux :

1.  L’iframe charge index.html
2.  Lecture du paramètre ?pdn=
3.  Téléchargement du fichier distant
4.  Affichage automatique

------------------------------------------------------------------------

**3. OPL *— PDN embarqué dans la page***

Page parent — PDN caché

    <div id="opl-pdn" style="color:#ffffff;font-size:1px;line-height:1px;">
    [Event "Tournoi"]
    [White "Blancs"]
    [Black "Noirs"]

    1. 34-30 16-21 2. 30-25 ...
    </div>

Page parent — Script postMessage

    <script>
    (function () {

      function getPDN() {
        const pdnBox = document.getElementById("opl-pdn");
        if (!pdnBox) return "";
        return (pdnBox.textContent || "").trim();
      }

      function sendPDN() {
        const iframe = document.getElementById("opl-frame");
        const pdn = getPDN();
        if (!iframe || !iframe.contentWindow || !pdn) return;

        iframe.contentWindow.postMessage(
          { type: "OPL_PDN", pdn: pdn },
          "*"
        );
      }

      window.addEventListener("message", function (e) {
        if (e.data && e.data.type === "OPL_READY") {
          sendPDN();
        }
      });

    })();
    </script>

Page parent — Iframe OPL

    <iframe
      id="opl-frame"
      src="https://votre-serveur/OPL.html"
      width="100%"
      height="650"
      style="border:1px solid #ccc;border-radius:10px;">
    </iframe>

Côté OPL.html — Envoi READY

    <script>
    window.parent.postMessage({ type: "OPL_READY" }, "*");
    </script>

Côté OPL.html — Réception PDN

    <script>
    window.addEventListener("message", function (e) {
      if (e.data && e.data.type === "OPL_PDN") {
        const pdn = e.data.pdn || "";
        // parser et afficher la partie
      }
    });
    </script>

------------------------------------------------------------------------

4. OPL2 — À venir

-   Multi-PDN sur une page
-   Sélecteur dynamique
-   Variantes arborescentes
-   Paramètre #ply
-   Sécurisation origin

------------------------------------------------------------------------
**🎯 Résumé**
Mode	Source PDN	
 - Mode Source PDN -Complexité Idéal pour index.html Local / Collé ⭐
 -   Simple Usage direct  SimplyPDN Fichier distant ⭐⭐ Très simple GitHub   / Codeberg 
  - OPL PDN embarqué page ⭐⭐⭐ Moyen SportsRégions     
 -   OPL2   Multi-PDN   avancé 🚧 À venir Portails complexes
Fin du README.
