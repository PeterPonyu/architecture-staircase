/* Consume the warehouse figure contract; never PeerJ FigureN.pdf paths. */
fetch("./data/figures.json")
  .catch(function () {
    return fetch("../papers/FIGURE-INDEX.json");
  })
  .then(function (response) {
    if (!response || !response.ok) {
      return null;
    }
    return response.json();
  })
  .then(function (index) {
    window.__FIGURE_INDEX__ = index || null;
  });
