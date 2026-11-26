# AlphaGo Zero, Live

A phone-friendly web page that teaches the AlphaGo Zero method
(Silver et al., 2017, *Nature* 550) with a mini AI that trains in the browser.

## Files
- `index.html`: page layout and styles (light-blue, dark honeycomb theme)
- `engine.js`: the AI: game rules, lookup-table "network", MCTS (PUCT), self-play training
- `app.js`: connects the AI to the page: boards, loop diagram, chart, search stepper, play mode

## Run it in VS Code
1. Open this folder in VS Code (File → Open Folder…).
2. Easiest: install the **Live Server** extension, right-click `index.html` → **Open with Live Server**.
   It reloads automatically when you save a file.
3. Or just double-click `index.html` to open it in your browser.
4. To see it as a phone: in Chrome press F12 → click the phone icon (device toolbar).

## Test the AI without the page (needs Node.js)
```
node -e "const A=require('./engine.js');const n=new A.Net();for(let g=0;g<400;g++)A.selfPlay(n,40);console.log('loses',A.vsPerfect(n,0,200))"
```
After about 300–500 self-play games, the network alone should almost never lose to a perfect player.

## Things to try
- `engine.js` → `selfPlay(net, 40)`: the 40 is simulations per move (the paper used 1,600).
- `Search` → `this.c = 1.5`: c_puct. Higher = more exploration.
- `SAMPLE_MOVES = 3`: how many opening moves are sampled for variety (temperature τ = 1).
- `Net` → `this.lr = 0.3`: learning rate.
- `index.html` → `--c:#5cc8ff`: the theme colour.
