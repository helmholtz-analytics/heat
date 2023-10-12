window.BENCHMARK_DATA = {
  "lastUpdate": 1697087000947,
  "repoUrl": "https://github.com/helmholtz-analytics/heat",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "juanpedroghm@gmail.com",
            "name": "JuanPedroGHM",
            "username": "JuanPedroGHM"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "02d2492d30515936f8c5345e049fd1e5fc005d61",
          "message": "Continous Benchmarking Action (#1137)\n\n* ci: continous benchmarking action\r\n\r\n* fix pytorch install command\r\n\r\n* fixed action python version\r\n\r\n* added github token to benchmark action\r\n\r\n* test message worst performance\r\n\r\n* ci: test gh-pages results from cb action\r\n\r\n* ci: this should pass\r\n\r\n* fix: typo\r\n\r\n* remove cache action in favor of gh-pages\r\n\r\n* extra /\r\n\r\n* ci: action config should work now\r\n\r\n* extra benchmarks from linalg module\r\n\r\n* quicker linalg benchmarks\r\n\r\n* cluster benchmarks, removed some code duplication\r\n\r\n* corrected cb workflow command\r\n\r\n* ci: split cb workflow into a main and pr. pr workflow triggers with a 'run bench' comment\r\n\r\n* ci: bench pr runs after a review is requested\r\n\r\n* ci: bench now only triggers on pull requests with 'PR talk' tag\r\n\r\n* ci: reshape benchmark, removed cronjob from pytorch workflows, renamed 'old' benchmarking folder to '2020'\r\n\r\n* mend\r\n\r\n* fix: missing import in cb main\r\n\r\n* ci: changed benchmark python version to 3.10, added shield in readme pointing to the benchmarks, changed trigger tag\r\n\r\n* fix: incorrect python version param\r\n\r\n---------\r\n\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>",
          "timestamp": "2023-04-27T05:11:16+02:00",
          "tree_id": "1c96d21a920a2e553af9ed5fe701a7203ef9e8b5",
          "url": "https://github.com/helmholtz-analytics/heat/commit/02d2492d30515936f8c5345e049fd1e5fc005d61"
        },
        "date": 1682566048655,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.0064384937286377,
            "unit": "s",
            "range": 0.06588414311408997
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.23874568939209,
            "unit": "s",
            "range": 4.7537031173706055
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 4.608992576599121,
            "unit": "s",
            "range": 1.0803240537643433
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 40.65354537963867,
            "unit": "s",
            "range": 2.2326583862304688
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.9593979120254517,
            "unit": "s",
            "range": 0.0668589398264885
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.9952268600463867,
            "unit": "s",
            "range": 0.05162478983402252
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 7.1640496253967285,
            "unit": "s",
            "range": 0.788973867893219
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 12.917398452758789,
            "unit": "s",
            "range": 0.052389681339263916
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m.tarnawa@fz-juelich.de",
            "name": "Michael Tarnawa",
            "username": "mtar"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7ca840102ee6ba51eb3db9c393dab973562f9cf1",
          "message": "Outsource CI workflow (main) (#1144)\n\n* change ci to public target repository\r\n\r\n* add more pytorch versions in matrix\r\n\r\n* delete gitlab file\r\n\r\n* Use gitlab's badge\r\n\r\n* sparse tests increase min pytorch to 1.12\r\n\r\n* update ref",
          "timestamp": "2023-05-08T09:47:07+02:00",
          "tree_id": "b47ac8cf8376639ad244c6cfa73a2d4439bf5fe9",
          "url": "https://github.com/helmholtz-analytics/heat/commit/7ca840102ee6ba51eb3db9c393dab973562f9cf1"
        },
        "date": 1683533022745,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.3607841730117798,
            "unit": "s",
            "range": 0.09720459580421448
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.596490859985352,
            "unit": "s",
            "range": 4.744739055633545
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.109837532043457,
            "unit": "s",
            "range": 0.9763903021812439
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 40.129493713378906,
            "unit": "s",
            "range": 2.6870570182800293
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 1.2492748498916626,
            "unit": "s",
            "range": 0.11195642501115799
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 1.1880028247833252,
            "unit": "s",
            "range": 0.11158335208892822
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 9.023608207702637,
            "unit": "s",
            "range": 0.6376816034317017
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 12.526630401611328,
            "unit": "s",
            "range": 0.2068168818950653
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39374113+ClaudiaComito@users.noreply.github.com",
            "name": "Claudia Comito",
            "username": "ClaudiaComito"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8abbc161454a10d02025b58365904f6be623409a",
          "message": "Features/560 reshape speedup (#1125)\n\n* reshape speed up first draft\r\n\r\n* lazy solution for non-zero split\r\n\r\n* remove dead code\r\n\r\n* remove test change\r\n\r\n* double precision for lshape_map, target_map\r\n\r\n* expand docs\r\n\r\n* prevent torch.prod() RuntimeError on older GPUs when input is int or long",
          "timestamp": "2023-05-22T04:50:40+02:00",
          "tree_id": "16f5a79210d34e878e1459356c5cb8c17122ea67",
          "url": "https://github.com/helmholtz-analytics/heat/commit/8abbc161454a10d02025b58365904f6be623409a"
        },
        "date": 1684724646853,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 0.9409732818603516,
            "unit": "s",
            "range": 0.10981494933366776
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 4.830145835876465,
            "unit": "s",
            "range": 3.8494036197662354
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 4.587499141693115,
            "unit": "s",
            "range": 1.0906234979629517
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 41.0870246887207,
            "unit": "s",
            "range": 2.2554943561553955
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.7591880559921265,
            "unit": "s",
            "range": 0.06596166640520096
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.7841414213180542,
            "unit": "s",
            "range": 0.08697084337472916
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.583443641662598,
            "unit": "s",
            "range": 0.5895178318023682
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.1486433744430542,
            "unit": "s",
            "range": 0.035412367433309555
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39374113+ClaudiaComito@users.noreply.github.com",
            "name": "Claudia Comito",
            "username": "ClaudiaComito"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6205a5eca137afaa5c19a1e7ccd2ee8a331329b0",
          "message": "Introduce `DNDarray.__array__()` method (#1154)\n\n* implement __array__ method\r\n\r\n* test __array__ method\r\n\r\n* test __array__ method\r\n\r\n* copy to CPU if necessary\r\n\r\n* update .numpy() docs\r\n\r\n* update .numpy() docs\r\n\r\n* update docs\r\n\r\n* update .numpy() docs\r\n\r\n* fix np dtype comparison\r\n\r\n* fix np dtype comparison\r\n\r\n* fix GPU array comparison",
          "timestamp": "2023-05-22T09:46:41+02:00",
          "tree_id": "76642d29fe5bbe4afafb9855c84eb0bd28add9de",
          "url": "https://github.com/helmholtz-analytics/heat/commit/6205a5eca137afaa5c19a1e7ccd2ee8a331329b0"
        },
        "date": 1684742390899,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 0.8627721667289734,
            "unit": "s",
            "range": 0.05123291537165642
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 4.71013879776001,
            "unit": "s",
            "range": 3.7869374752044678
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 4.644545555114746,
            "unit": "s",
            "range": 0.9805055856704712
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 40.26018142700195,
            "unit": "s",
            "range": 1.976609230041504
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.8286346197128296,
            "unit": "s",
            "range": 0.06730403751134872
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.7945326566696167,
            "unit": "s",
            "range": 0.05571514740586281
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.884799957275391,
            "unit": "s",
            "range": 0.5157168507575989
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.1258480548858643,
            "unit": "s",
            "range": 0.05847302824258804
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m.tarnawa@fz-juelich.de",
            "name": "Michael Tarnawa",
            "username": "mtar"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f400d14b7324c2f7180eb8ae7c2070de066135eb",
          "message": "Merge pull request #1152 from helmholtz-analytics/support/694-amd-hip\n\nsupport amd HIP testing",
          "timestamp": "2023-05-22T11:21:10+02:00",
          "tree_id": "3660c17358ac7b03d42d170fd2c00b414f79b302",
          "url": "https://github.com/helmholtz-analytics/heat/commit/f400d14b7324c2f7180eb8ae7c2070de066135eb"
        },
        "date": 1684748047091,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 0.8988556861877441,
            "unit": "s",
            "range": 0.0637301653623581
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.219430446624756,
            "unit": "s",
            "range": 4.604112148284912
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 4.559315204620361,
            "unit": "s",
            "range": 0.9846170544624329
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 38.10189437866211,
            "unit": "s",
            "range": 2.76328444480896
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.810444176197052,
            "unit": "s",
            "range": 0.06511379033327103
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.8347999453544617,
            "unit": "s",
            "range": 0.026931846514344215
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.731255531311035,
            "unit": "s",
            "range": 0.7050207257270813
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.0273901224136353,
            "unit": "s",
            "range": 0.03521817550063133
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m.tarnawa@fz-juelich.de",
            "name": "Michael Tarnawa",
            "username": "mtar"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f99cce1453e161ef400cbe1119ac6fef80dfd1fa",
          "message": "Drop support for PyTorch 1.7, Python 3.6 (#1147)\n\n* increase min pytorch & delete old code\r\n\r\n* delete code for python < 3.7\r\n\r\n* remove pytorch 1.7 from test matrix\r\n\r\n* call bitwise functions instead of tensor builtins\r\n\r\n* replace python 3.7 with 3.10 in tests\r\n\r\n* increase min python + numpy\r\n\r\n* update deprecated functions\r\n\r\n* Place 3.10 in quotes so it doesn't get interpreted as 3.1\r\n\r\n* Update ci.yaml\r\n\r\nexclude python 3.10 and pytorch 1.8/1.9\r\n\r\n* Update ci.yaml\r\n\r\nExclude Python 3.10 & PyTorch 1.10\r\n\r\n---------\r\n\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>",
          "timestamp": "2023-05-23T10:34:33+02:00",
          "tree_id": "daef152187a22241c34bfac8e8373136ae55535b",
          "url": "https://github.com/helmholtz-analytics/heat/commit/f99cce1453e161ef400cbe1119ac6fef80dfd1fa"
        },
        "date": 1684831714798,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.0331406593322754,
            "unit": "s",
            "range": 0.06368827074766159
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.409095764160156,
            "unit": "s",
            "range": 4.7657294273376465
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 4.845440864562988,
            "unit": "s",
            "range": 1.0106961727142334
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 39.47422409057617,
            "unit": "s",
            "range": 2.9162349700927734
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.951638400554657,
            "unit": "s",
            "range": 0.06195865571498871
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 1.0053038597106934,
            "unit": "s",
            "range": 0.05874303728342056
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.7009596824646,
            "unit": "s",
            "range": 0.5610595941543579
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.3282134532928467,
            "unit": "s",
            "range": 0.032579205930233
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39374113+ClaudiaComito@users.noreply.github.com",
            "name": "Claudia Comito",
            "username": "ClaudiaComito"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "468999e108a72485b3688f54c8d7fe2f984f8045",
          "message": "Support PyTorch 2.0.1 on branch `main` (#1155)\n\n* Support latest PyTorch release\r\n\r\n* Update setup.py\r\n\r\n* Specify allclose tolerance in test_inv()\r\n\r\n* Increase allclose tolerance in test_inv\r\n\r\n* Increase allclose tolerance for distributed floating-point operations\r\n\r\n* fix working branches selection\r\n\r\n* add pr workflow (#1127)\r\n\r\n* add pr workflow\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* Update README.md\r\n\r\n* Update .gitlab-ci.yml\r\n\r\n---------\r\n\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\n\r\n* Support latest PyTorch release\r\n\r\n* expand version check to torch 2\r\n\r\n* dndarray.item() to return ValueError if dndarray.size > 1\r\n\r\n* increase min pytorch & delete old code\r\n\r\n* delete code for python < 3.7\r\n\r\n* remove pytorch 1.7 from test matrix\r\n\r\n* call bitwise functions instead of tensor builtins\r\n\r\n* Fixed bugs due to behaviour change in torch.nn.Module.zero_grad() and torch.Tensor.item() in torch 2.0 (#1149)\r\n\r\n* Support latest PyTorch release\r\n\r\n* Fixed bug in optimizer due to change in default behaviour of zero_grad() function in torch 2.0\r\n\r\n* Fixed bug in due to change in behaviour of torch.Tensor.item() in torch 2.0\r\n\r\n* Fixed version in setup.py\r\n\r\n* Modified tests to work for torch 1.0 also\r\n\r\n---------\r\n\r\nCo-authored-by: ClaudiaComito <ClaudiaComito@users.noreply.github.com>\r\n\r\n* update test_item\r\n\r\n* replace python 3.7 with 3.10 in tests\r\n\r\n* increase min python + numpy\r\n\r\n* Add Interoperability chapter\r\n\r\n* update deprecated functions\r\n\r\n* switch ref\r\n\r\n* bump pytorch version\r\n\r\n* define branch name for Pytorch release PR\r\n\r\n* Add torch 2.0.0 to matrix\r\n\r\n* Test latest torch patch, fix 1.8 entries\r\n\r\n---------\r\n\r\nCo-authored-by: ClaudiaComito <ClaudiaComito@users.noreply.github.com>\r\nCo-authored-by: Michael Tarnawa <m.tarnawa@fz-juelich.de>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: Ashwath V A <73862377+Mystic-Slice@users.noreply.github.com>",
          "timestamp": "2023-05-23T19:26:43+02:00",
          "tree_id": "4298fe3897568cd95d835ffc2d9cc53b298fb3fe",
          "url": "https://github.com/helmholtz-analytics/heat/commit/468999e108a72485b3688f54c8d7fe2f984f8045"
        },
        "date": 1684863622471,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.3457703590393066,
            "unit": "s",
            "range": 0.0818895697593689
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.620911598205566,
            "unit": "s",
            "range": 4.693271636962891
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.0497331619262695,
            "unit": "s",
            "range": 0.9303812980651855
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 40.066925048828125,
            "unit": "s",
            "range": 2.2695720195770264
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.8463083505630493,
            "unit": "s",
            "range": 0.05460250750184059
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.8126780390739441,
            "unit": "s",
            "range": 0.048105787485837936
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 7.060009002685547,
            "unit": "s",
            "range": 0.18940185010433197
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.1943029165267944,
            "unit": "s",
            "range": 0.060206275433301926
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "112093564+mrfh92@users.noreply.github.com",
            "name": "Fabian Hoppe",
            "username": "mrfh92"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5fd3af5e6ba8c12b89a3bb80ee1c05ae49a4de5b",
          "message": "Merge pull request #1075 from AsRaNi1/Communicator-not-properly-initialized-when-creating-new-DNDarrays-in-some-routines/1074-my-bug-fix\n\nFixed initialization of DNDarrays communicator in some routines",
          "timestamp": "2023-05-25T11:28:04+02:00",
          "tree_id": "0fefe822e3ad13c426a22230d176be12246185ab",
          "url": "https://github.com/helmholtz-analytics/heat/commit/5fd3af5e6ba8c12b89a3bb80ee1c05ae49a4de5b"
        },
        "date": 1685007717392,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.4233367443084717,
            "unit": "s",
            "range": 0.10170768201351166
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.2173333168029785,
            "unit": "s",
            "range": 3.9342546463012695
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.108536720275879,
            "unit": "s",
            "range": 1.0454715490341187
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 40.01653289794922,
            "unit": "s",
            "range": 2.342420816421509
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.8153241276741028,
            "unit": "s",
            "range": 0.038919467478990555
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.8794156908988953,
            "unit": "s",
            "range": 0.08604833483695984
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.73513650894165,
            "unit": "s",
            "range": 0.6942393183708191
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.2570745944976807,
            "unit": "s",
            "range": 0.04073786735534668
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "112093564+mrfh92@users.noreply.github.com",
            "name": "Fabian Hoppe",
            "username": "mrfh92"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1bc1ccaa27bb030f2b04931ce2dc6dbae055942e",
          "message": "Merge pull request #1126 from helmholtz-analytics/features/1041-hsvd-new\n\nadded implementation of hSVD (`heat.linalg.hsvd`, hierarchical SVD) allowing for truncation both by rank (`heat.linalg.hsvd_rank`) and accuracy (`heat.linalg.hsvd_rtol`)",
          "timestamp": "2023-06-06T13:04:10+02:00",
          "tree_id": "26c9d8b1f7b3677f137fce88515bea192f96d961",
          "url": "https://github.com/helmholtz-analytics/heat/commit/1bc1ccaa27bb030f2b04931ce2dc6dbae055942e"
        },
        "date": 1686050263087,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.20815221965312958,
            "unit": "s",
            "range": 0.01585559733211994
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.27276965975761414,
            "unit": "s",
            "range": 0.05782872438430786
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.3340585231781006,
            "unit": "s",
            "range": 0.0755491629242897
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.560429096221924,
            "unit": "s",
            "range": 4.55959415435791
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.081908226013184,
            "unit": "s",
            "range": 1.1351022720336914
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 40.4936408996582,
            "unit": "s",
            "range": 1.4693344831466675
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.7979428172111511,
            "unit": "s",
            "range": 0.040488023310899734
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.8364278674125671,
            "unit": "s",
            "range": 0.052982114255428314
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.182063102722168,
            "unit": "s",
            "range": 0.7953183650970459
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.1357619762420654,
            "unit": "s",
            "range": 0.06439079344272614
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "112093564+mrfh92@users.noreply.github.com",
            "name": "Fabian Hoppe",
            "username": "mrfh92"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "90166e49912051f7e75475c76a72e52dfcf2dd80",
          "message": "Merge pull request #1020 from helmholtz-analytics/feature/778-broadcasting\n\nImplement `broadcast_arrays`, `broadcast_to`",
          "timestamp": "2023-06-15T14:47:27+02:00",
          "tree_id": "4a052b5653f767e32985ee0a37d36f682be3c12c",
          "url": "https://github.com/helmholtz-analytics/heat/commit/90166e49912051f7e75475c76a72e52dfcf2dd80"
        },
        "date": 1686834185225,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.27992305159568787,
            "unit": "s",
            "range": 0.023444192484021187
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.3306518793106079,
            "unit": "s",
            "range": 0.049682874232530594
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.9918584823608398,
            "unit": "s",
            "range": 0.12343338131904602
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.967545509338379,
            "unit": "s",
            "range": 4.010085105895996
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.619624137878418,
            "unit": "s",
            "range": 0.9951204657554626
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 39.32096862792969,
            "unit": "s",
            "range": 3.0978775024414062
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 1.2608810663223267,
            "unit": "s",
            "range": 0.08409810066223145
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 1.3107084035873413,
            "unit": "s",
            "range": 0.08187070488929749
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 9.585488319396973,
            "unit": "s",
            "range": 0.7710549235343933
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.401508092880249,
            "unit": "s",
            "range": 0.06584632396697998
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "112093564+mrfh92@users.noreply.github.com",
            "name": "Fabian Hoppe",
            "username": "mrfh92"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3a840ccc25ef06fb97056b5ca3a57cb7cf17d062",
          "message": "Merge pull request #1119 from helmholtz-analytics/features/#1117-array-copy-None\n\n`ht.array()` default to `copy=None` (e.g., only if necessary)",
          "timestamp": "2023-06-19T14:55:48+02:00",
          "tree_id": "c817366808f53be844d30abed6a3f6e2634c700a",
          "url": "https://github.com/helmholtz-analytics/heat/commit/3a840ccc25ef06fb97056b5ca3a57cb7cf17d062"
        },
        "date": 1687180181267,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.22918958961963654,
            "unit": "s",
            "range": 0.025974951684474945
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.2832033038139343,
            "unit": "s",
            "range": 0.053751759231090546
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.1935369968414307,
            "unit": "s",
            "range": 0.06838201731443405
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.231768608093262,
            "unit": "s",
            "range": 4.600335597991943
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.160141944885254,
            "unit": "s",
            "range": 1.168915033340454
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 41.266937255859375,
            "unit": "s",
            "range": 2.0074045658111572
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.8308317065238953,
            "unit": "s",
            "range": 0.04742327332496643
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.842497706413269,
            "unit": "s",
            "range": 0.057537686079740524
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.376896858215332,
            "unit": "s",
            "range": 0.6989758610725403
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.0855686664581299,
            "unit": "s",
            "range": 0.04543221369385719
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "sai.suraj.27.729@gmail.com",
            "name": "Sai-Suraj-27",
            "username": "Sai-Suraj-27"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9ea256b0085808aa5db50320b11114d08a9373e8",
          "message": "Refactoring for efficiency and readability (#1150)\n\n* Reformatted some code in a few files to make it more cleaner.\r\n\r\n* Used f-strings whereever they can be used but not used, also merged some nested if conditions.\r\n\r\n* Used f-strings whereever they can be used but not used, also merged some nested if conditions.\r\n\r\n* updated conf.py file by using f-strings at required places.\r\n\r\n* Reformatted some code in 4 more files, to make it better.\r\n\r\n* Updated more files by using f-strings wherever required.\r\n\r\n* Updated some more files to make them more cleaner.\r\n\r\n* Reformatted some code in 5 more files, to make it better.\r\n\r\n* Made the code a little cleaner in 3 more files.\r\n\r\n* Reformatted some code in 6 more files, to make it better.\r\n\r\n* Updated 5 more files to make them a little more cleaner and better.\r\n\r\n* Reformatted some code in 6 more files, to make them better.\r\n\r\n* Updated 5 more files to make them a little more cleaner and better.\r\n\r\n* Updated a few more files to make their code much cleaner and better.\r\n\r\n* Fixed errors in pre-commit checks.\r\n\r\n* Updated the code in 2 files as per the suggested changes.\r\n\r\n* Made changes to resolve the errors in tests.\r\n\r\n* corrected a small error in test_knn.py file.\r\n\r\n* Reverted the small changes made in test_knn.py\r\n\r\n* Made the suggested changes to test_knn.py file.\r\n\r\n* Update test_knn.py\r\n\r\n* Undoing the changes made in test_knn.py file.\r\n\r\n* Changed the dtype from ht.int32 to ht.int64 in test_knn.py file.\r\n\r\n---------\r\n\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>",
          "timestamp": "2023-06-20T13:50:33+02:00",
          "tree_id": "c8cfd580478e896907f7167b8bf91de098a6a307",
          "url": "https://github.com/helmholtz-analytics/heat/commit/9ea256b0085808aa5db50320b11114d08a9373e8"
        },
        "date": 1687262705181,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.219100683927536,
            "unit": "s",
            "range": 0.01044752448797226
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.3291913866996765,
            "unit": "s",
            "range": 0.05191640555858612
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.5395084619522095,
            "unit": "s",
            "range": 0.0899520292878151
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.364592552185059,
            "unit": "s",
            "range": 4.4310102462768555
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.169771671295166,
            "unit": "s",
            "range": 1.1930651664733887
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 39.88731384277344,
            "unit": "s",
            "range": 2.326538562774658
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.9160690307617188,
            "unit": "s",
            "range": 0.056772567331790924
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.9194231033325195,
            "unit": "s",
            "range": 0.06780950725078583
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 7.410874843597412,
            "unit": "s",
            "range": 0.4855055809020996
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.2732521295547485,
            "unit": "s",
            "range": 0.04698130860924721
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "b.hagemeier@fz-juelich.de",
            "name": "Björn Hagemeier",
            "username": "bhagemeier"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "966a7a840872b0f7e201c240bc45e2384921e2ce",
          "message": "Dockerfile and accompanying documentation (#970)\n\n* Dockerfile and accompanying documentation\r\n\r\nThe Dockerfile provides some flexibility in selecting which version of HeAT should be inside\r\nthe Docker image. Also, one can choose whether to install from source or from PyPI.\r\n\r\n* README.md describing containerization\r\n\r\n* Fix indentation in README.md\r\n\r\nSome code sections had a mix of spaces and tabs, which have now been\r\nconvertd into tabs.\r\n\r\n* Docker support\r\n\r\nUse pytorch 1.11\r\nFix problem with CUDA package repo keys\r\n\r\n* Ensure mpi4py installation from source\r\n\r\n* Migrate to NVidia PyTorch base image\r\n\r\nNVidia images come with support for HPC systems desirable for our uses.\r\nThey work a little differently internally and required some changes.\r\n\r\nThe tzdata configuration configures the CET/CEST timezone, which seems\r\nto be required when installing additional packages.\r\n\r\nThere is an issue with pip caches in the image, which led to the final\r\ncache purge to fail in the PyPI release based build. This is fixed\r\nthrough a final invocation of true.\r\n\r\n* Provide sample file for Singularity\r\n\r\n* feat: singularity definition file and slurm multi-node example in the docker readme\r\n\r\n* docs: quick_start.md has a docker section with link to docker readme\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* ci: docker cleanup\r\n\r\n* ci: build docker action, updated docs\r\n\r\n* Apply suggestions from code review\r\n\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>\r\n\r\n* README suggestions\r\n\r\n* docs: removed system specific flag from example slurm file\r\n\r\n---------\r\n\r\nCo-authored-by: Gutiérrez Hermosillo Muriedas, Juan Pedro <juanpedroghm@gmail.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>",
          "timestamp": "2023-06-20T15:02:15+02:00",
          "tree_id": "b5d148d6866ec7989df95ce86a2d42b0f190e16d",
          "url": "https://github.com/helmholtz-analytics/heat/commit/966a7a840872b0f7e201c240bc45e2384921e2ce"
        },
        "date": 1687266951204,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.22041860222816467,
            "unit": "s",
            "range": 0.024018704891204834
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.2794017493724823,
            "unit": "s",
            "range": 0.03424530103802681
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.2342188358306885,
            "unit": "s",
            "range": 0.051946934312582016
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.46388578414917,
            "unit": "s",
            "range": 4.5065388679504395
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.0080976486206055,
            "unit": "s",
            "range": 1.0073717832565308
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 39.77582931518555,
            "unit": "s",
            "range": 2.422520637512207
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.7612134218215942,
            "unit": "s",
            "range": 0.0695384070277214
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.7703195214271545,
            "unit": "s",
            "range": 0.04877445101737976
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.442999362945557,
            "unit": "s",
            "range": 0.668419599533081
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.0710986852645874,
            "unit": "s",
            "range": 0.07150645554065704
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39374113+ClaudiaComito@users.noreply.github.com",
            "name": "Claudia Comito",
            "username": "ClaudiaComito"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f98ccc6eaa298a77a65fd7847d8f1f04691e4fc2",
          "message": "update version after release of v1.3.0 (#1167)",
          "timestamp": "2023-06-20T18:59:56+02:00",
          "tree_id": "c8795b5046495f76052214b82419fc5fc7df6d0b",
          "url": "https://github.com/helmholtz-analytics/heat/commit/f98ccc6eaa298a77a65fd7847d8f1f04691e4fc2"
        },
        "date": 1687281227116,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.22124862670898438,
            "unit": "s",
            "range": 0.030435005202889442
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.27973777055740356,
            "unit": "s",
            "range": 0.04762342944741249
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.358171820640564,
            "unit": "s",
            "range": 0.0987866148352623
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.0979509353637695,
            "unit": "s",
            "range": 3.916285514831543
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 4.819827079772949,
            "unit": "s",
            "range": 1.0077662467956543
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 39.493404388427734,
            "unit": "s",
            "range": 2.39079213142395
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.7739747166633606,
            "unit": "s",
            "range": 0.03930717706680298
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.8921969532966614,
            "unit": "s",
            "range": 0.05291367322206497
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.896584510803223,
            "unit": "s",
            "range": 0.6321520209312439
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.1826903820037842,
            "unit": "s",
            "range": 0.033491428941488266
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "112093564+mrfh92@users.noreply.github.com",
            "name": "Fabian Hoppe",
            "username": "mrfh92"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e15a2d3c1cadedf9b0ad04d3625f739929e9e64a",
          "message": "Merge pull request #1166 from helmholtz-analytics/docs/release_md\n\nDocumentation: Release HowTo",
          "timestamp": "2023-06-26T11:59:45+02:00",
          "tree_id": "094c8bee1c1824e4ebaba27565a191842415dc03",
          "url": "https://github.com/helmholtz-analytics/heat/commit/e15a2d3c1cadedf9b0ad04d3625f739929e9e64a"
        },
        "date": 1687774554352,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.3064005672931671,
            "unit": "s",
            "range": 0.023807071149349213
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.37893012166023254,
            "unit": "s",
            "range": 0.07647167146205902
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 2.102416753768921,
            "unit": "s",
            "range": 0.11848271638154984
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 6.573990821838379,
            "unit": "s",
            "range": 4.802183151245117
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.885861396789551,
            "unit": "s",
            "range": 1.0264192819595337
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 40.816734313964844,
            "unit": "s",
            "range": 1.3686072826385498
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 1.2656538486480713,
            "unit": "s",
            "range": 0.08347979933023453
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 1.2928130626678467,
            "unit": "s",
            "range": 0.09839753806591034
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 10.350786209106445,
            "unit": "s",
            "range": 0.18308517336845398
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.3596116304397583,
            "unit": "s",
            "range": 0.06540459394454956
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "112093564+mrfh92@users.noreply.github.com",
            "name": "Fabian Hoppe",
            "username": "mrfh92"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "405d9b985c131f80afe33b1d2b2bd8125f97cf62",
          "message": "Merge pull request #1173 from helmholtz-analytics/fix/changelog-updater\n\nChangelog updater action fix",
          "timestamp": "2023-07-03T13:46:58+02:00",
          "tree_id": "403ee60b1197c55de786761d54762bbe63dba349",
          "url": "https://github.com/helmholtz-analytics/heat/commit/405d9b985c131f80afe33b1d2b2bd8125f97cf62"
        },
        "date": 1688385686491,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.23451419174671173,
            "unit": "s",
            "range": 0.023987354710698128
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.29232892394065857,
            "unit": "s",
            "range": 0.0527816116809845
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.3277208805084229,
            "unit": "s",
            "range": 0.0941959023475647
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.213129043579102,
            "unit": "s",
            "range": 4.287837982177734
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.16896915435791,
            "unit": "s",
            "range": 1.0000478029251099
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 42.04335403442383,
            "unit": "s",
            "range": 0.4768431782722473
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.8573762774467468,
            "unit": "s",
            "range": 0.056673262268304825
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.8971904516220093,
            "unit": "s",
            "range": 0.06855850666761398
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 7.060193061828613,
            "unit": "s",
            "range": 0.21249920129776
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.2614878416061401,
            "unit": "s",
            "range": 0.04088260605931282
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m.tarnawa@fz-juelich.de",
            "name": "Michael Tarnawa",
            "username": "mtar"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2b2ed0ff726e034ee2f77148d9c77bfe19c128dc",
          "message": "Merge pull request #1180 from helmholtz-analytics/pre-commit-ci-update-config\n\n[pre-commit.ci] pre-commit autoupdate",
          "timestamp": "2023-07-14T10:49:32+02:00",
          "tree_id": "c5bc964335117fcf0cc799cbb72fc22d216363a6",
          "url": "https://github.com/helmholtz-analytics/heat/commit/2b2ed0ff726e034ee2f77148d9c77bfe19c128dc"
        },
        "date": 1689325456461,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.2634623646736145,
            "unit": "s",
            "range": 0.023072421550750732
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.3210631012916565,
            "unit": "s",
            "range": 0.05724715441465378
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.5168782472610474,
            "unit": "s",
            "range": 0.09760332852602005
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.728908538818359,
            "unit": "s",
            "range": 4.325509548187256
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.005774021148682,
            "unit": "s",
            "range": 1.0516462326049805
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 39.976829528808594,
            "unit": "s",
            "range": 1.9820195436477661
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.9788303375244141,
            "unit": "s",
            "range": 0.04834107309579849
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.9679139852523804,
            "unit": "s",
            "range": 0.04410133883357048
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 7.1697869300842285,
            "unit": "s",
            "range": 0.682013988494873
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.2115697860717773,
            "unit": "s",
            "range": 0.048984088003635406
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "112093564+mrfh92@users.noreply.github.com",
            "name": "Fabian Hoppe",
            "username": "mrfh92"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "009a91aba5d68b7dc52121949b3e505f965fa765",
          "message": "Merge pull request #1170 from helmholtz-analytics/bug/1121-print-fails-on-gpu\n\n`ht.print` can now print arrays distributed over `n>1` GPUs",
          "timestamp": "2023-07-24T10:22:43+02:00",
          "tree_id": "83fc134312ed9b857f6c8a755ebedab2fb9e0087",
          "url": "https://github.com/helmholtz-analytics/heat/commit/009a91aba5d68b7dc52121949b3e505f965fa765"
        },
        "date": 1690187794006,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.24136164784431458,
            "unit": "s",
            "range": 0.014989631250500679
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.3060407042503357,
            "unit": "s",
            "range": 0.04474097862839699
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.3213037252426147,
            "unit": "s",
            "range": 0.07807199656963348
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.438258171081543,
            "unit": "s",
            "range": 4.532651901245117
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 4.776575088500977,
            "unit": "s",
            "range": 1.0958740711212158
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 39.36615753173828,
            "unit": "s",
            "range": 2.6294941902160645
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.8351085782051086,
            "unit": "s",
            "range": 0.04033607244491577
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.9164949655532837,
            "unit": "s",
            "range": 0.07127095013856888
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.665530204772949,
            "unit": "s",
            "range": 0.8261171579360962
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.1730554103851318,
            "unit": "s",
            "range": 0.048380810767412186
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39374113+ClaudiaComito@users.noreply.github.com",
            "name": "Claudia Comito",
            "username": "ClaudiaComito"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "90ac056447197b82a94e644aab179f0ef1f1c1a6",
          "message": "Automate branch creation on issue assignment (#1181)\n\n* automate branch creation on issue assignment\r\n\r\n* add enhancement label to feature request issues",
          "timestamp": "2023-07-24T15:00:07+02:00",
          "tree_id": "827f9dc84cdce34e9d611d3d58a71e77606d513b",
          "url": "https://github.com/helmholtz-analytics/heat/commit/90ac056447197b82a94e644aab179f0ef1f1c1a6"
        },
        "date": 1690204502587,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.28999075293540955,
            "unit": "s",
            "range": 0.030202291905879974
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.3314424455165863,
            "unit": "s",
            "range": 0.05256454274058342
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.4538590908050537,
            "unit": "s",
            "range": 0.09711333364248276
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.473354816436768,
            "unit": "s",
            "range": 3.811626672744751
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.401356220245361,
            "unit": "s",
            "range": 1.0672577619552612
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 41.215797424316406,
            "unit": "s",
            "range": 1.128321886062622
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.877597451210022,
            "unit": "s",
            "range": 0.05254720523953438
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.9881324768066406,
            "unit": "s",
            "range": 0.04088718816637993
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 7.017214298248291,
            "unit": "s",
            "range": 0.7237522602081299
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.2957892417907715,
            "unit": "s",
            "range": 0.036403100937604904
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39374113+ClaudiaComito@users.noreply.github.com",
            "name": "Claudia Comito",
            "username": "ClaudiaComito"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1524a761015ae169f94794d5719e171e8dccf7b3",
          "message": "Merge latest 1.3.x updates into `main` (#1188)\n\n* update version to 1.3.0\r\n\r\n* Create .readthedocs.yaml (#1187)\r\n\r\n* Update heat/core/version.py\r\n\r\n---------\r\n\r\nCo-authored-by: Michael Tarnawa <m.tarnawa@fz-juelich.de>",
          "timestamp": "2023-08-04T11:08:35+02:00",
          "tree_id": "2113f1a6014b646b469f02f2a21004ec1eb5ab5d",
          "url": "https://github.com/helmholtz-analytics/heat/commit/1524a761015ae169f94794d5719e171e8dccf7b3"
        },
        "date": 1691141052930,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.26202335953712463,
            "unit": "s",
            "range": 0.024828922003507614
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.3398238718509674,
            "unit": "s",
            "range": 0.04830550774931908
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 2.0454630851745605,
            "unit": "s",
            "range": 0.10846072435379028
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 6.077889442443848,
            "unit": "s",
            "range": 3.951051950454712
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.871243476867676,
            "unit": "s",
            "range": 1.0575283765792847
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 41.26012420654297,
            "unit": "s",
            "range": 1.60458505153656
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 1.152302861213684,
            "unit": "s",
            "range": 0.04583846777677536
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 1.1569578647613525,
            "unit": "s",
            "range": 0.06399905681610107
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 9.02987289428711,
            "unit": "s",
            "range": 0.5764942765235901
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.3018343448638916,
            "unit": "s",
            "range": 0.061232734471559525
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39374113+ClaudiaComito@users.noreply.github.com",
            "name": "Claudia Comito",
            "username": "ClaudiaComito"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "549deb9c7e7732696c5b7a96bbd8f01c5bd03f53",
          "message": "Small fixes to automated branch creation on Issue assignment (#1190)\n\n* small changes to branch creation action, update docs\r\n\r\n* update quick_start\r\n\r\n* Update quick_start.md",
          "timestamp": "2023-08-04T14:51:17+02:00",
          "tree_id": "981b44c24ca3e79a3ea00c361f9c8773a64f9e6d",
          "url": "https://github.com/helmholtz-analytics/heat/commit/549deb9c7e7732696c5b7a96bbd8f01c5bd03f53"
        },
        "date": 1691154436762,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.27335071563720703,
            "unit": "s",
            "range": 0.034357961267232895
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.33771198987960815,
            "unit": "s",
            "range": 0.046306051313877106
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.9356781244277954,
            "unit": "s",
            "range": 0.12699483335018158
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 6.227564811706543,
            "unit": "s",
            "range": 4.617743968963623
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.638108730316162,
            "unit": "s",
            "range": 1.0548932552337646
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 39.32331085205078,
            "unit": "s",
            "range": 2.513068437576294
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 1.2127825021743774,
            "unit": "s",
            "range": 0.08655709028244019
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 1.3259118795394897,
            "unit": "s",
            "range": 0.08576948195695877
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 9.925756454467773,
            "unit": "s",
            "range": 0.29191234707832336
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.3343473672866821,
            "unit": "s",
            "range": 0.0658719390630722
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juanpedroghm@gmail.com",
            "name": "JuanPedroGHM",
            "username": "JuanPedroGHM"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7482f14dca5574d054b3eba47070c33e49f6ad74",
          "message": "FAIR-RS and best practice badges (#1143)\n\n* ci: fair software workflow\r\n\r\n* Update README.md: added FAIR badge as required by FAIR action\r\n\r\n* Update README.md: added zenodo/doi badge\r\n\r\n* Update README.md: corrected zenodo/doi badge to all-version-doi\r\n\r\n* Update README.md: added pypi badge\r\n\r\n* Update README.md: correction\r\n\r\n* Trigger workflow\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* Add OpenSSF Best Practices badge\r\n\r\n* docs: rearrange badges by color, removed fair-software action\r\n\r\n* reorganize / relabel badges\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* Update README.md\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* Test - Update README.md\r\n\r\n* Update README.md\r\n\r\n---------\r\n\r\nCo-authored-by: Fabian Hoppe <112093564+mrfh92@users.noreply.github.com>\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>",
          "timestamp": "2023-08-07T10:16:57+02:00",
          "tree_id": "595ced4f79fad949f12a565566f52ea727a96b70",
          "url": "https://github.com/helmholtz-analytics/heat/commit/7482f14dca5574d054b3eba47070c33e49f6ad74"
        },
        "date": 1691397089780,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.21091003715991974,
            "unit": "s",
            "range": 0.014791554771363735
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.292398065328598,
            "unit": "s",
            "range": 0.04492587968707085
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.3392932415008545,
            "unit": "s",
            "range": 0.09163504093885422
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 5.540284156799316,
            "unit": "s",
            "range": 4.387272357940674
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 5.125346660614014,
            "unit": "s",
            "range": 0.9564374685287476
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 41.636573791503906,
            "unit": "s",
            "range": 0.7615768909454346
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.8575516939163208,
            "unit": "s",
            "range": 0.07045251131057739
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.8904763460159302,
            "unit": "s",
            "range": 0.05301974341273308
          },
          {
            "name": "qr_split_01_N4_CPU - RUNTIME",
            "value": 6.388546466827393,
            "unit": "s",
            "range": 0.8297274112701416
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 1.164061188697815,
            "unit": "s",
            "range": 0.06034034118056297
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Abdul Samad Siddiqui",
            "username": "samadpls",
            "email": "abdulsamadsid1@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ba32e21e35cea5c358a11c4125a9f395fcc87fee",
          "message": "Implement `to_sparse()` method for DNDarray conversion to DCSR_matrix (#1206)\n\n* Implemented `to_sparse()` method for DNDarray class conversion to DCSR_matrix representation\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* added `to_sparse` in heat/sparse/manipulations.py\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* added for `to_sparse` method\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* removed comment\r\n\r\n* updated testcase of `to_sparse`\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* updated `to_sparse` method and test case\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* Updated the `todense` method to `to_dense`\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n---------\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>",
          "timestamp": "2023-09-18T15:41:38Z",
          "url": "https://github.com/helmholtz-analytics/heat/commit/ba32e21e35cea5c358a11c4125a9f395fcc87fee"
        },
        "date": 1695061210788,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "heat_benchmarks_N1_GPU - RUNTIME",
            "value": 54.71043014526367,
            "unit": "s",
            "range": 11.475862503051758
          },
          {
            "name": "heat_benchmarks_N1_GPU - ENERGY",
            "value": 5.657191787320312,
            "unit": "kJ",
            "range": 1.461198985442853
          },
          {
            "name": "matmul_split_0_N1_GPU - RUNTIME",
            "value": 0.003910933621227741,
            "unit": "s",
            "range": 0.010307220742106438
          },
          {
            "name": "matmul_split_0_N1_GPU - POWER",
            "value": 57.234453830658936,
            "unit": "W",
            "range": 7.170973812883382
          },
          {
            "name": "matmul_split_0_N1_GPU - CPU_UTIL",
            "value": 64.85218691739588,
            "unit": "%",
            "range": 10.820350038482214
          },
          {
            "name": "matmul_split_0_N1_GPU - GPU_UTIL",
            "value": 23.240501499176027,
            "unit": "%",
            "range": 25.38449333243275
          },
          {
            "name": "matmul_split_1_N1_GPU - RUNTIME",
            "value": 0.00030682608485221863,
            "unit": "s",
            "range": 0.00009590132685843855
          },
          {
            "name": "matmul_split_1_N1_GPU - POWER",
            "value": 57.1877993780427,
            "unit": "W",
            "range": 7.0338160897787585
          },
          {
            "name": "matmul_split_1_N1_GPU - CPU_UTIL",
            "value": 64.3101242625594,
            "unit": "%",
            "range": 11.024419317086444
          },
          {
            "name": "matmul_split_1_N1_GPU - GPU_UTIL",
            "value": 23.2497145652771,
            "unit": "%",
            "range": 25.379120994391712
          },
          {
            "name": "qr_split_0_N1_GPU - RUNTIME",
            "value": 0.43145403265953064,
            "unit": "s",
            "range": 0.12630599737167358
          },
          {
            "name": "qr_split_0_N1_GPU - POWER",
            "value": 56.883765256939895,
            "unit": "W",
            "range": 6.240057844906999
          },
          {
            "name": "qr_split_0_N1_GPU - CPU_UTIL",
            "value": 60.40641895043566,
            "unit": "%",
            "range": 12.393361712388407
          },
          {
            "name": "qr_split_0_N1_GPU - GPU_UTIL",
            "value": 23.316209316253662,
            "unit": "%",
            "range": 25.341206657657786
          },
          {
            "name": "qr_split_1_N1_GPU - RUNTIME",
            "value": 0.39341962337493896,
            "unit": "s",
            "range": 0.10198221355676651
          },
          {
            "name": "qr_split_1_N1_GPU - POWER",
            "value": 56.40486210182051,
            "unit": "W",
            "range": 5.054335809164441
          },
          {
            "name": "qr_split_1_N1_GPU - CPU_UTIL",
            "value": 53.86511748416289,
            "unit": "%",
            "range": 15.878604525046596
          },
          {
            "name": "qr_split_1_N1_GPU - GPU_UTIL",
            "value": 23.411893558502197,
            "unit": "%",
            "range": 25.289310572531146
          },
          {
            "name": "lanczos_N1_GPU - RUNTIME",
            "value": 0.7607436180114746,
            "unit": "s",
            "range": 0.2596266269683838
          },
          {
            "name": "lanczos_N1_GPU - POWER",
            "value": 56.08505449294216,
            "unit": "W",
            "range": 3.99273759640158
          },
          {
            "name": "lanczos_N1_GPU - CPU_UTIL",
            "value": 50.02803881255871,
            "unit": "%",
            "range": 18.677442985741674
          },
          {
            "name": "lanczos_N1_GPU - GPU_UTIL",
            "value": 23.42926025390625,
            "unit": "%",
            "range": 25.280229490121634
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - RUNTIME",
            "value": 0.09880927950143814,
            "unit": "s",
            "range": 0.029307741671800613
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - POWER",
            "value": 56.00516091470645,
            "unit": "W",
            "range": 3.4529581507739375
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - CPU_UTIL",
            "value": 49.577965596180285,
            "unit": "%",
            "range": 19.253951307104018
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - GPU_UTIL",
            "value": 23.429584121704103,
            "unit": "%",
            "range": 25.28006112783634
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - RUNTIME",
            "value": 0.17148229479789734,
            "unit": "s",
            "range": 0.04275234788656235
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - POWER",
            "value": 56.04492287120735,
            "unit": "W",
            "range": 3.2144861569707004
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - CPU_UTIL",
            "value": 49.69602286102173,
            "unit": "%",
            "range": 19.23323167868713
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - GPU_UTIL",
            "value": 23.42977695465088,
            "unit": "%",
            "range": 25.279960901064296
          },
          {
            "name": "kmeans_N1_GPU - RUNTIME",
            "value": 8.592199325561523,
            "unit": "s",
            "range": 1.8948616981506348
          },
          {
            "name": "kmeans_N1_GPU - POWER",
            "value": 99.71591692414653,
            "unit": "W",
            "range": 9.03606982118179
          },
          {
            "name": "kmeans_N1_GPU - CPU_UTIL",
            "value": 50.98196783491269,
            "unit": "%",
            "range": 18.16128179907022
          },
          {
            "name": "kmeans_N1_GPU - GPU_UTIL",
            "value": 23.511600685119628,
            "unit": "%",
            "range": 25.19734086398459
          },
          {
            "name": "kmedians_N1_GPU - RUNTIME",
            "value": 20.76929473876953,
            "unit": "s",
            "range": 3.4454426765441895
          },
          {
            "name": "kmedians_N1_GPU - POWER",
            "value": 100.54395933048289,
            "unit": "W",
            "range": 7.571087873902628
          },
          {
            "name": "kmedians_N1_GPU - CPU_UTIL",
            "value": 47.78072894024875,
            "unit": "%",
            "range": 16.692631343089833
          },
          {
            "name": "kmedians_N1_GPU - GPU_UTIL",
            "value": 23.543073654174805,
            "unit": "%",
            "range": 25.24054186681957
          },
          {
            "name": "kmedoids_N1_GPU - RUNTIME",
            "value": 22.940574645996094,
            "unit": "s",
            "range": 6.851184368133545
          },
          {
            "name": "kmedoids_N1_GPU - POWER",
            "value": 99.846030011377,
            "unit": "W",
            "range": 8.589105115976112
          },
          {
            "name": "kmedoids_N1_GPU - CPU_UTIL",
            "value": 46.65680722665907,
            "unit": "%",
            "range": 17.10585955240965
          },
          {
            "name": "kmedoids_N1_GPU - GPU_UTIL",
            "value": 21.54621343612671,
            "unit": "%",
            "range": 21.857382590383782
          },
          {
            "name": "reshape_N1_GPU - RUNTIME",
            "value": 0.0005241393810138106,
            "unit": "s",
            "range": 0.00020541498088277876
          },
          {
            "name": "reshape_N1_GPU - POWER",
            "value": 66.69323187876572,
            "unit": "W",
            "range": 5.143763437808746
          },
          {
            "name": "reshape_N1_GPU - CPU_UTIL",
            "value": 47.60529801301992,
            "unit": "%",
            "range": 16.225618834758098
          },
          {
            "name": "reshape_N1_GPU - GPU_UTIL",
            "value": 23.20434503555298,
            "unit": "%",
            "range": 25.40545164519388
          },
          {
            "name": "concatenate_N1_GPU - RUNTIME",
            "value": 0.001865386962890625,
            "unit": "s",
            "range": 0.0007046397076919675
          },
          {
            "name": "concatenate_N1_GPU - POWER",
            "value": 66.7922160512455,
            "unit": "W",
            "range": 5.143354542553551
          },
          {
            "name": "concatenate_N1_GPU - CPU_UTIL",
            "value": 47.59613667616065,
            "unit": "%",
            "range": 16.23079958230375
          },
          {
            "name": "concatenate_N1_GPU - GPU_UTIL",
            "value": 23.205159378051757,
            "unit": "%",
            "range": 25.405181570797055
          },
          {
            "name": "heat_benchmarks_N4_CPU - RUNTIME",
            "value": 18.48904037475586,
            "unit": "s",
            "range": 1.4508240222930908
          },
          {
            "name": "heat_benchmarks_N4_CPU - ENERGY",
            "value": 1.931828287100976,
            "unit": "kJ",
            "range": 0.3574017166139596
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.7019883394241333,
            "unit": "s",
            "range": 0.10119219124317169
          },
          {
            "name": "matmul_split_0_N4_CPU - POWER",
            "value": 30.060594199891682,
            "unit": "W",
            "range": 20.93693072823485
          },
          {
            "name": "matmul_split_0_N4_CPU - CPU_UTIL",
            "value": 97.13432622669663,
            "unit": "%",
            "range": 0.5274541877536626
          },
          {
            "name": "matmul_split_0_N4_CPU - GPU_UTIL",
            "value": 4.89044189453125,
            "unit": "%",
            "range": 1.6563371599403582
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.6995142102241516,
            "unit": "s",
            "range": 0.13847310841083527
          },
          {
            "name": "matmul_split_1_N4_CPU - POWER",
            "value": 30.17897136816374,
            "unit": "W",
            "range": 21.085202922994142
          },
          {
            "name": "matmul_split_1_N4_CPU - CPU_UTIL",
            "value": 97.1518138996189,
            "unit": "%",
            "range": 0.7813571918076159
          },
          {
            "name": "matmul_split_1_N4_CPU - GPU_UTIL",
            "value": 4.89044189453125,
            "unit": "%",
            "range": 1.6563371599403582
          },
          {
            "name": "qr_split_0_N4_CPU - RUNTIME",
            "value": 3.4768409729003906,
            "unit": "s",
            "range": 0.3260354697704315
          },
          {
            "name": "qr_split_0_N4_CPU - POWER",
            "value": 80.81578495097497,
            "unit": "W",
            "range": 17.04901873474495
          },
          {
            "name": "qr_split_0_N4_CPU - CPU_UTIL",
            "value": 96.90982591250463,
            "unit": "%",
            "range": 1.0722373223642208
          },
          {
            "name": "qr_split_0_N4_CPU - GPU_UTIL",
            "value": 5.037067264318466,
            "unit": "%",
            "range": 1.3073253213550864
          },
          {
            "name": "qr_split_1_N4_CPU - RUNTIME",
            "value": 3.2145614624023438,
            "unit": "s",
            "range": 0.3068188428878784
          },
          {
            "name": "qr_split_1_N4_CPU - POWER",
            "value": 84.0908898493755,
            "unit": "W",
            "range": 22.56126607543422
          },
          {
            "name": "qr_split_1_N4_CPU - CPU_UTIL",
            "value": 96.99470566618632,
            "unit": "%",
            "range": 1.2083298284952761
          },
          {
            "name": "qr_split_1_N4_CPU - GPU_UTIL",
            "value": 5.174127769470215,
            "unit": "%",
            "range": 1.302150938141108
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 1.2840831279754639,
            "unit": "s",
            "range": 0.17169275879859924
          },
          {
            "name": "lanczos_N4_CPU - POWER",
            "value": 57.396504206313054,
            "unit": "W",
            "range": 27.729228486878398
          },
          {
            "name": "lanczos_N4_CPU - CPU_UTIL",
            "value": 96.98874356745719,
            "unit": "%",
            "range": 1.0646583457035117
          },
          {
            "name": "lanczos_N4_CPU - GPU_UTIL",
            "value": 5.07232666015625,
            "unit": "%",
            "range": 1.6816792766103308
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.23114462196826935,
            "unit": "s",
            "range": 0.04563005641102791
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - POWER",
            "value": 36.81581965669728,
            "unit": "W",
            "range": 18.63671162082194
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - CPU_UTIL",
            "value": 96.97656748063926,
            "unit": "%",
            "range": 1.1433128122438865
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - GPU_UTIL",
            "value": 5.07232666015625,
            "unit": "%",
            "range": 1.6816792766103308
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.2722078561782837,
            "unit": "s",
            "range": 0.049168143421411514
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - POWER",
            "value": 36.26996587949584,
            "unit": "W",
            "range": 18.909633819833576
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - CPU_UTIL",
            "value": 97.05218851937434,
            "unit": "%",
            "range": 1.0818455830474538
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - GPU_UTIL",
            "value": 5.07232666015625,
            "unit": "%",
            "range": 1.6816792766103308
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.7023112773895264,
            "unit": "s",
            "range": 0.1437886506319046
          },
          {
            "name": "kmeans_N4_CPU - POWER",
            "value": 63.50597037756207,
            "unit": "W",
            "range": 32.29619715878747
          },
          {
            "name": "kmeans_N4_CPU - CPU_UTIL",
            "value": 96.97738077192679,
            "unit": "%",
            "range": 0.8422842850047717
          },
          {
            "name": "kmeans_N4_CPU - GPU_UTIL",
            "value": 5.085903978347778,
            "unit": "%",
            "range": 1.6908491078151044
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 2.1626739501953125,
            "unit": "s",
            "range": 0.2502463161945343
          },
          {
            "name": "kmedians_N4_CPU - POWER",
            "value": 70.53155490740322,
            "unit": "W",
            "range": 22.262869295415104
          },
          {
            "name": "kmedians_N4_CPU - CPU_UTIL",
            "value": 96.99289542903793,
            "unit": "%",
            "range": 1.0429797127938734
          },
          {
            "name": "kmedians_N4_CPU - GPU_UTIL",
            "value": 5.136292374134063,
            "unit": "%",
            "range": 1.7318570615103162
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 1.997444748878479,
            "unit": "s",
            "range": 0.15550798177719116
          },
          {
            "name": "kmedoids_N4_CPU - POWER",
            "value": 66.12241001263683,
            "unit": "W",
            "range": 26.009890983875916
          },
          {
            "name": "kmedoids_N4_CPU - CPU_UTIL",
            "value": 97.1178940445758,
            "unit": "%",
            "range": 0.8930285064772991
          },
          {
            "name": "kmedoids_N4_CPU - GPU_UTIL",
            "value": 5.184482836723328,
            "unit": "%",
            "range": 1.7824180877209461
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 0.7350853681564331,
            "unit": "s",
            "range": 0.046704091131687164
          },
          {
            "name": "reshape_N4_CPU - POWER",
            "value": 34.17005385874796,
            "unit": "W",
            "range": 20.4758913721985
          },
          {
            "name": "reshape_N4_CPU - CPU_UTIL",
            "value": 96.89440085076832,
            "unit": "%",
            "range": 0.7983750688521047
          },
          {
            "name": "reshape_N4_CPU - GPU_UTIL",
            "value": 5.19683837890625,
            "unit": "%",
            "range": 1.797026337594576
          },
          {
            "name": "concatenate_N4_CPU - RUNTIME",
            "value": 0.7041054964065552,
            "unit": "s",
            "range": 0.07449857145547867
          },
          {
            "name": "concatenate_N4_CPU - POWER",
            "value": 45.06281707521747,
            "unit": "W",
            "range": 26.558343240630737
          },
          {
            "name": "concatenate_N4_CPU - CPU_UTIL",
            "value": 96.34261269323169,
            "unit": "%",
            "range": 1.7528627511787767
          },
          {
            "name": "concatenate_N4_CPU - GPU_UTIL",
            "value": 5.19683837890625,
            "unit": "%",
            "range": 1.797026337594576
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Abdul Samad Siddiqui",
            "username": "samadpls",
            "email": "abdulsamadsid1@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ba32e21e35cea5c358a11c4125a9f395fcc87fee",
          "message": "Implement `to_sparse()` method for DNDarray conversion to DCSR_matrix (#1206)\n\n* Implemented `to_sparse()` method for DNDarray class conversion to DCSR_matrix representation\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* added `to_sparse` in heat/sparse/manipulations.py\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* added for `to_sparse` method\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* removed comment\r\n\r\n* updated testcase of `to_sparse`\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* updated `to_sparse` method and test case\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* Updated the `todense` method to `to_dense`\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n---------\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>",
          "timestamp": "2023-09-18T15:41:38Z",
          "url": "https://github.com/helmholtz-analytics/heat/commit/ba32e21e35cea5c358a11c4125a9f395fcc87fee"
        },
        "date": 1695062576631,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "heat_benchmarks_N1_GPU - RUNTIME",
            "value": 59.69823455810547,
            "unit": "s",
            "range": 15.331486701965332
          },
          {
            "name": "heat_benchmarks_N1_GPU - ENERGY",
            "value": 6.241065937773438,
            "unit": "kJ",
            "range": 1.6312500771596055
          },
          {
            "name": "matmul_split_0_N1_GPU - RUNTIME",
            "value": 0.0031237986404448748,
            "unit": "s",
            "range": 0.00800480879843235
          },
          {
            "name": "matmul_split_0_N1_GPU - POWER",
            "value": 55.2111372072636,
            "unit": "W",
            "range": 4.34461011875589
          },
          {
            "name": "matmul_split_0_N1_GPU - CPU_UTIL",
            "value": 48.67094619260912,
            "unit": "%",
            "range": 1.6119278841434947
          },
          {
            "name": "matmul_split_0_N1_GPU - GPU_UTIL",
            "value": 24.45414810180664,
            "unit": "%",
            "range": 24.071585402396966
          },
          {
            "name": "matmul_split_1_N1_GPU - RUNTIME",
            "value": 0.0003100503236055374,
            "unit": "s",
            "range": 0.00007620583346579224
          },
          {
            "name": "matmul_split_1_N1_GPU - POWER",
            "value": 55.39044476981519,
            "unit": "W",
            "range": 4.414555405728455
          },
          {
            "name": "matmul_split_1_N1_GPU - CPU_UTIL",
            "value": 48.77921324289251,
            "unit": "%",
            "range": 1.702927460500752
          },
          {
            "name": "matmul_split_1_N1_GPU - GPU_UTIL",
            "value": 24.464479827880858,
            "unit": "%",
            "range": 24.07615984858132
          },
          {
            "name": "qr_split_0_N1_GPU - RUNTIME",
            "value": 0.4983418583869934,
            "unit": "s",
            "range": 0.2266666740179062
          },
          {
            "name": "qr_split_0_N1_GPU - POWER",
            "value": 57.19442458975385,
            "unit": "W",
            "range": 5.939843233447779
          },
          {
            "name": "qr_split_0_N1_GPU - CPU_UTIL",
            "value": 49.58148887367112,
            "unit": "%",
            "range": 4.053320017411083
          },
          {
            "name": "qr_split_0_N1_GPU - GPU_UTIL",
            "value": 24.578219985961915,
            "unit": "%",
            "range": 24.129093770553887
          },
          {
            "name": "qr_split_1_N1_GPU - RUNTIME",
            "value": 0.48739710450172424,
            "unit": "s",
            "range": 0.22840580344200134
          },
          {
            "name": "qr_split_1_N1_GPU - POWER",
            "value": 58.6136977806441,
            "unit": "W",
            "range": 6.908505506950354
          },
          {
            "name": "qr_split_1_N1_GPU - CPU_UTIL",
            "value": 51.759064417788935,
            "unit": "%",
            "range": 8.088525987704472
          },
          {
            "name": "qr_split_1_N1_GPU - GPU_UTIL",
            "value": 24.631195068359375,
            "unit": "%",
            "range": 24.15535378736739
          },
          {
            "name": "lanczos_N1_GPU - RUNTIME",
            "value": 0.8479291200637817,
            "unit": "s",
            "range": 0.2943582534790039
          },
          {
            "name": "lanczos_N1_GPU - POWER",
            "value": 58.09249185260846,
            "unit": "W",
            "range": 5.739692162802088
          },
          {
            "name": "lanczos_N1_GPU - CPU_UTIL",
            "value": 53.39828313617359,
            "unit": "%",
            "range": 11.353520526688198
          },
          {
            "name": "lanczos_N1_GPU - GPU_UTIL",
            "value": 24.631195068359375,
            "unit": "%",
            "range": 24.15535378736739
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - RUNTIME",
            "value": 0.10327472537755966,
            "unit": "s",
            "range": 0.03640708699822426
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - POWER",
            "value": 58.50793566084555,
            "unit": "W",
            "range": 7.659988059176985
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - CPU_UTIL",
            "value": 53.732824215505524,
            "unit": "%",
            "range": 12.873338942531475
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - GPU_UTIL",
            "value": 24.631210708618163,
            "unit": "%",
            "range": 24.155361690505835
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - RUNTIME",
            "value": 0.18937084078788757,
            "unit": "s",
            "range": 0.03811651095747948
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - POWER",
            "value": 58.22457577839894,
            "unit": "W",
            "range": 6.653556623899675
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - CPU_UTIL",
            "value": 53.2337861836847,
            "unit": "%",
            "range": 12.708122049383318
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - GPU_UTIL",
            "value": 24.63135986328125,
            "unit": "%",
            "range": 24.15543706390947
          },
          {
            "name": "kmeans_N1_GPU - RUNTIME",
            "value": 9.470705032348633,
            "unit": "s",
            "range": 2.304272413253784
          },
          {
            "name": "kmeans_N1_GPU - POWER",
            "value": 98.30856427674956,
            "unit": "W",
            "range": 9.2310162919553
          },
          {
            "name": "kmeans_N1_GPU - CPU_UTIL",
            "value": 51.780512339005284,
            "unit": "%",
            "range": 13.1199932162943
          },
          {
            "name": "kmeans_N1_GPU - GPU_UTIL",
            "value": 24.561867237091064,
            "unit": "%",
            "range": 24.263124349712033
          },
          {
            "name": "kmedians_N1_GPU - RUNTIME",
            "value": 23.08001708984375,
            "unit": "s",
            "range": 7.165721893310547
          },
          {
            "name": "kmedians_N1_GPU - POWER",
            "value": 101.66237149301656,
            "unit": "W",
            "range": 7.7163394456358985
          },
          {
            "name": "kmedians_N1_GPU - CPU_UTIL",
            "value": 49.615828469487454,
            "unit": "%",
            "range": 11.571616823390189
          },
          {
            "name": "kmedians_N1_GPU - GPU_UTIL",
            "value": 21.87959976196289,
            "unit": "%",
            "range": 22.572452734848856
          },
          {
            "name": "kmedoids_N1_GPU - RUNTIME",
            "value": 24.32750129699707,
            "unit": "s",
            "range": 7.378664493560791
          },
          {
            "name": "kmedoids_N1_GPU - POWER",
            "value": 103.92407625705957,
            "unit": "W",
            "range": 8.984703187122287
          },
          {
            "name": "kmedoids_N1_GPU - CPU_UTIL",
            "value": 54.60465694990504,
            "unit": "%",
            "range": 14.347066422230219
          },
          {
            "name": "kmedoids_N1_GPU - GPU_UTIL",
            "value": 25.399933052062988,
            "unit": "%",
            "range": 23.862381146778166
          },
          {
            "name": "reshape_N1_GPU - RUNTIME",
            "value": 0.00048065185546875,
            "unit": "s",
            "range": 0.00018628069665282965
          },
          {
            "name": "reshape_N1_GPU - POWER",
            "value": 66.47844843522859,
            "unit": "W",
            "range": 4.765265227425529
          },
          {
            "name": "reshape_N1_GPU - CPU_UTIL",
            "value": 52.72451815062912,
            "unit": "%",
            "range": 17.05758422559628
          },
          {
            "name": "reshape_N1_GPU - GPU_UTIL",
            "value": 28.67687225341797,
            "unit": "%",
            "range": 28.809653346555102
          },
          {
            "name": "concatenate_N1_GPU - RUNTIME",
            "value": 0.0019294738303869963,
            "unit": "s",
            "range": 0.0008731987909413874
          },
          {
            "name": "concatenate_N1_GPU - POWER",
            "value": 66.55699149929885,
            "unit": "W",
            "range": 4.757026535514324
          },
          {
            "name": "concatenate_N1_GPU - CPU_UTIL",
            "value": 52.7249261392519,
            "unit": "%",
            "range": 17.06123274038441
          },
          {
            "name": "concatenate_N1_GPU - GPU_UTIL",
            "value": 28.67878007888794,
            "unit": "%",
            "range": 28.808524353790006
          },
          {
            "name": "heat_benchmarks_N4_CPU - RUNTIME",
            "value": 18.128376007080078,
            "unit": "s",
            "range": 1.413793921470642
          },
          {
            "name": "heat_benchmarks_N4_CPU - ENERGY",
            "value": 1.7995661830914063,
            "unit": "kJ",
            "range": 0.19804026575663092
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.6256526708602905,
            "unit": "s",
            "range": 0.07142762094736099
          },
          {
            "name": "matmul_split_0_N4_CPU - POWER",
            "value": 48.785960127452874,
            "unit": "W",
            "range": 5.332805909889873
          },
          {
            "name": "matmul_split_0_N4_CPU - CPU_UTIL",
            "value": 59.01791548425334,
            "unit": "%",
            "range": 9.048809888062454
          },
          {
            "name": "matmul_split_0_N4_CPU - GPU_UTIL",
            "value": 3.5170959293842317,
            "unit": "%",
            "range": 0.5829342811623265
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.537792444229126,
            "unit": "s",
            "range": 0.047536153346300125
          },
          {
            "name": "matmul_split_1_N4_CPU - POWER",
            "value": 48.859842873805775,
            "unit": "W",
            "range": 5.265617239231247
          },
          {
            "name": "matmul_split_1_N4_CPU - CPU_UTIL",
            "value": 63.11189882773347,
            "unit": "%",
            "range": 13.74644772652
          },
          {
            "name": "matmul_split_1_N4_CPU - GPU_UTIL",
            "value": 3.5506019473075865,
            "unit": "%",
            "range": 0.6025847036299482
          },
          {
            "name": "qr_split_0_N4_CPU - RUNTIME",
            "value": 3.153442859649658,
            "unit": "s",
            "range": 0.26641395688056946
          },
          {
            "name": "qr_split_0_N4_CPU - POWER",
            "value": 84.17563568465475,
            "unit": "W",
            "range": 4.744364543261209
          },
          {
            "name": "qr_split_0_N4_CPU - CPU_UTIL",
            "value": 62.32423938465049,
            "unit": "%",
            "range": 13.425434667246256
          },
          {
            "name": "qr_split_0_N4_CPU - GPU_UTIL",
            "value": 3.5522321224212647,
            "unit": "%",
            "range": 0.7386599530925063
          },
          {
            "name": "qr_split_1_N4_CPU - RUNTIME",
            "value": 3.150428295135498,
            "unit": "s",
            "range": 0.2635411024093628
          },
          {
            "name": "qr_split_1_N4_CPU - POWER",
            "value": 82.27374083963994,
            "unit": "W",
            "range": 11.367913704762016
          },
          {
            "name": "qr_split_1_N4_CPU - CPU_UTIL",
            "value": 61.92006668340027,
            "unit": "%",
            "range": 11.288893709825814
          },
          {
            "name": "qr_split_1_N4_CPU - GPU_UTIL",
            "value": 3.46436585187912,
            "unit": "%",
            "range": 0.9785401769232376
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 1.367933750152588,
            "unit": "s",
            "range": 0.4469675123691559
          },
          {
            "name": "lanczos_N4_CPU - POWER",
            "value": 61.243289713574995,
            "unit": "W",
            "range": 19.558260369961932
          },
          {
            "name": "lanczos_N4_CPU - CPU_UTIL",
            "value": 62.045913769713,
            "unit": "%",
            "range": 8.223872346459768
          },
          {
            "name": "lanczos_N4_CPU - GPU_UTIL",
            "value": 3.3607835292816164,
            "unit": "%",
            "range": 1.0117738279714124
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.21000508964061737,
            "unit": "s",
            "range": 0.035516273230314255
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - POWER",
            "value": 46.511918525603726,
            "unit": "W",
            "range": 11.642097886282782
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - CPU_UTIL",
            "value": 62.43130146265092,
            "unit": "%",
            "range": 11.461568365731779
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - GPU_UTIL",
            "value": 3.32377353310585,
            "unit": "%",
            "range": 1.0174054006037963
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.2189672291278839,
            "unit": "s",
            "range": 0.0507146455347538
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - POWER",
            "value": 46.17824627422769,
            "unit": "W",
            "range": 12.604349922552679
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - CPU_UTIL",
            "value": 62.48116477033061,
            "unit": "%",
            "range": 12.539928965545768
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - GPU_UTIL",
            "value": 3.3228431046009064,
            "unit": "%",
            "range": 1.0178839557767823
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.978151559829712,
            "unit": "s",
            "range": 0.2906028926372528
          },
          {
            "name": "kmeans_N4_CPU - POWER",
            "value": 67.77175148982414,
            "unit": "W",
            "range": 20.291054854641953
          },
          {
            "name": "kmeans_N4_CPU - CPU_UTIL",
            "value": 62.799614598059485,
            "unit": "%",
            "range": 11.585053667634046
          },
          {
            "name": "kmeans_N4_CPU - GPU_UTIL",
            "value": 3.346914863586426,
            "unit": "%",
            "range": 1.0157436235415276
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 2.1712939739227295,
            "unit": "s",
            "range": 0.5137801766395569
          },
          {
            "name": "kmedians_N4_CPU - POWER",
            "value": 70.41121639652788,
            "unit": "W",
            "range": 13.182674913550493
          },
          {
            "name": "kmedians_N4_CPU - CPU_UTIL",
            "value": 63.541066511504596,
            "unit": "%",
            "range": 12.957058689740625
          },
          {
            "name": "kmedians_N4_CPU - GPU_UTIL",
            "value": 3.4009456813335417,
            "unit": "%",
            "range": 1.040512257233945
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 1.909646987915039,
            "unit": "s",
            "range": 0.38621556758880615
          },
          {
            "name": "kmedoids_N4_CPU - POWER",
            "value": 76.53245166726603,
            "unit": "W",
            "range": 21.410376257543355
          },
          {
            "name": "kmedoids_N4_CPU - CPU_UTIL",
            "value": 65.58474331105712,
            "unit": "%",
            "range": 16.93855786002926
          },
          {
            "name": "kmedoids_N4_CPU - GPU_UTIL",
            "value": 4.957687616348267,
            "unit": "%",
            "range": 4.8408860541146
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 0.7445775866508484,
            "unit": "s",
            "range": 0.1402420550584793
          },
          {
            "name": "reshape_N4_CPU - POWER",
            "value": 50.43356415157923,
            "unit": "W",
            "range": 4.862415941089486
          },
          {
            "name": "reshape_N4_CPU - CPU_UTIL",
            "value": 65.36576546855173,
            "unit": "%",
            "range": 14.12322565389997
          },
          {
            "name": "reshape_N4_CPU - GPU_UTIL",
            "value": 6.469639283418656,
            "unit": "%",
            "range": 9.001491003366823
          },
          {
            "name": "concatenate_N4_CPU - RUNTIME",
            "value": 0.8419097661972046,
            "unit": "s",
            "range": 0.21008077263832092
          },
          {
            "name": "concatenate_N4_CPU - POWER",
            "value": 53.919723468281816,
            "unit": "W",
            "range": 10.797210238027304
          },
          {
            "name": "concatenate_N4_CPU - CPU_UTIL",
            "value": 64.09882308563905,
            "unit": "%",
            "range": 12.07656190021263
          },
          {
            "name": "concatenate_N4_CPU - GPU_UTIL",
            "value": 6.548939615488052,
            "unit": "%",
            "range": 8.974715018266213
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Abdul Samad Siddiqui",
            "username": "samadpls",
            "email": "abdulsamadsid1@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ba32e21e35cea5c358a11c4125a9f395fcc87fee",
          "message": "Implement `to_sparse()` method for DNDarray conversion to DCSR_matrix (#1206)\n\n* Implemented `to_sparse()` method for DNDarray class conversion to DCSR_matrix representation\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* added `to_sparse` in heat/sparse/manipulations.py\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* added for `to_sparse` method\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* removed comment\r\n\r\n* updated testcase of `to_sparse`\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* updated `to_sparse` method and test case\r\n\r\n* [pre-commit.ci] auto fixes from pre-commit.com hooks\r\n\r\nfor more information, see https://pre-commit.ci\r\n\r\n* Updated the `todense` method to `to_dense`\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\n\r\n---------\r\n\r\nSigned-off-by: samadpls <abdulsamadsid1@gmail.com>\r\nCo-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>",
          "timestamp": "2023-09-18T15:41:38Z",
          "url": "https://github.com/helmholtz-analytics/heat/commit/ba32e21e35cea5c358a11c4125a9f395fcc87fee"
        },
        "date": 1695311484666,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "heat_benchmarks_N1_GPU - RUNTIME",
            "value": 48.723541259765625,
            "unit": "s",
            "range": 3.56638240814209
          },
          {
            "name": "heat_benchmarks_N1_GPU - ENERGY",
            "value": 4.54576955179375,
            "unit": "kJ",
            "range": 0.24823885637210344
          },
          {
            "name": "matmul_split_0_N1_GPU - RUNTIME",
            "value": 0.002965368330478668,
            "unit": "s",
            "range": 0.0069899181835353374
          },
          {
            "name": "matmul_split_0_N1_GPU - POWER",
            "value": 45.386964757093416,
            "unit": "W",
            "range": 7.344680853355894
          },
          {
            "name": "matmul_split_0_N1_GPU - CPU_UTIL",
            "value": 44.109082552245084,
            "unit": "%",
            "range": 0.6866004456240025
          },
          {
            "name": "matmul_split_0_N1_GPU - GPU_UTIL",
            "value": 7.909654724597931,
            "unit": "%",
            "range": 2.182476091037054
          },
          {
            "name": "matmul_split_1_N1_GPU - RUNTIME",
            "value": 0.00038596391095779836,
            "unit": "s",
            "range": 0.00015771028120070696
          },
          {
            "name": "matmul_split_1_N1_GPU - POWER",
            "value": 45.613630435072096,
            "unit": "W",
            "range": 6.853414107617037
          },
          {
            "name": "matmul_split_1_N1_GPU - CPU_UTIL",
            "value": 44.1322396974998,
            "unit": "%",
            "range": 0.7443681498845156
          },
          {
            "name": "matmul_split_1_N1_GPU - GPU_UTIL",
            "value": 7.917329490184784,
            "unit": "%",
            "range": 2.154081164564621
          },
          {
            "name": "qr_split_0_N1_GPU - RUNTIME",
            "value": 0.34009990096092224,
            "unit": "s",
            "range": 0.07294216006994247
          },
          {
            "name": "qr_split_0_N1_GPU - POWER",
            "value": 46.77709590185073,
            "unit": "W",
            "range": 4.501383753512067
          },
          {
            "name": "qr_split_0_N1_GPU - CPU_UTIL",
            "value": 44.19114817937302,
            "unit": "%",
            "range": 1.6192989711880903
          },
          {
            "name": "qr_split_0_N1_GPU - GPU_UTIL",
            "value": 7.95026044845581,
            "unit": "%",
            "range": 2.0140746192495356
          },
          {
            "name": "qr_split_1_N1_GPU - RUNTIME",
            "value": 0.33036574721336365,
            "unit": "s",
            "range": 0.07074509561061859
          },
          {
            "name": "qr_split_1_N1_GPU - POWER",
            "value": 49.209933147925426,
            "unit": "W",
            "range": 1.950923663786184
          },
          {
            "name": "qr_split_1_N1_GPU - CPU_UTIL",
            "value": 44.41085321388985,
            "unit": "%",
            "range": 3.876843297235677
          },
          {
            "name": "qr_split_1_N1_GPU - GPU_UTIL",
            "value": 8.035554504394531,
            "unit": "%",
            "range": 1.707659685501281
          },
          {
            "name": "lanczos_N1_GPU - RUNTIME",
            "value": 0.6201338768005371,
            "unit": "s",
            "range": 0.15231482684612274
          },
          {
            "name": "lanczos_N1_GPU - POWER",
            "value": 50.05566658307765,
            "unit": "W",
            "range": 2.2077495756888568
          },
          {
            "name": "lanczos_N1_GPU - CPU_UTIL",
            "value": 44.728782257675824,
            "unit": "%",
            "range": 6.245812565777519
          },
          {
            "name": "lanczos_N1_GPU - GPU_UTIL",
            "value": 8.051160144805909,
            "unit": "%",
            "range": 1.606249894374883
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - RUNTIME",
            "value": 0.0912630707025528,
            "unit": "s",
            "range": 0.02884185127913952
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - POWER",
            "value": 49.39704776133887,
            "unit": "W",
            "range": 2.00179741756441
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - CPU_UTIL",
            "value": 44.84418910753379,
            "unit": "%",
            "range": 7.125258690075768
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - GPU_UTIL",
            "value": 8.044711089134216,
            "unit": "%",
            "range": 1.6028827833096335
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - RUNTIME",
            "value": 0.14667680859565735,
            "unit": "s",
            "range": 0.030560001730918884
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - POWER",
            "value": 49.42721948446303,
            "unit": "W",
            "range": 2.157399192605302
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - CPU_UTIL",
            "value": 44.81085485831651,
            "unit": "%",
            "range": 7.248165705869683
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - GPU_UTIL",
            "value": 8.043806314468384,
            "unit": "%",
            "range": 1.6021170016786186
          },
          {
            "name": "kmeans_N1_GPU - RUNTIME",
            "value": 7.82305908203125,
            "unit": "s",
            "range": 1.0188162326812744
          },
          {
            "name": "kmeans_N1_GPU - POWER",
            "value": 89.93934648200948,
            "unit": "W",
            "range": 4.651845762145068
          },
          {
            "name": "kmeans_N1_GPU - CPU_UTIL",
            "value": 43.206834227170305,
            "unit": "%",
            "range": 5.8252059381361745
          },
          {
            "name": "kmeans_N1_GPU - GPU_UTIL",
            "value": 7.99903290271759,
            "unit": "%",
            "range": 1.5601186004579748
          },
          {
            "name": "kmedians_N1_GPU - RUNTIME",
            "value": 19.023006439208984,
            "unit": "s",
            "range": 1.8236854076385498
          },
          {
            "name": "kmedians_N1_GPU - POWER",
            "value": 91.59222047949616,
            "unit": "W",
            "range": 2.446746131507208
          },
          {
            "name": "kmedians_N1_GPU - CPU_UTIL",
            "value": 44.671950274532264,
            "unit": "%",
            "range": 5.030511338605211
          },
          {
            "name": "kmedians_N1_GPU - GPU_UTIL",
            "value": 8.072515869140625,
            "unit": "%",
            "range": 1.5895754127895534
          },
          {
            "name": "kmedoids_N1_GPU - RUNTIME",
            "value": 19.768274307250977,
            "unit": "s",
            "range": 1.5539631843566895
          },
          {
            "name": "kmedoids_N1_GPU - POWER",
            "value": 90.51619829676608,
            "unit": "W",
            "range": 2.160491297453752
          },
          {
            "name": "kmedoids_N1_GPU - CPU_UTIL",
            "value": 43.88663120679458,
            "unit": "%",
            "range": 4.274663138696822
          },
          {
            "name": "kmedoids_N1_GPU - GPU_UTIL",
            "value": 8.270869994163514,
            "unit": "%",
            "range": 1.6222972112820455
          },
          {
            "name": "reshape_N1_GPU - RUNTIME",
            "value": 0.0004875183221884072,
            "unit": "s",
            "range": 0.0001999993110075593
          },
          {
            "name": "reshape_N1_GPU - POWER",
            "value": 58.817168666048964,
            "unit": "W",
            "range": 3.584710595983174
          },
          {
            "name": "reshape_N1_GPU - CPU_UTIL",
            "value": 45.14270970218337,
            "unit": "%",
            "range": 7.493804088840927
          },
          {
            "name": "reshape_N1_GPU - GPU_UTIL",
            "value": 8.69036145210266,
            "unit": "%",
            "range": 0.6818284926691671
          },
          {
            "name": "concatenate_N1_GPU - RUNTIME",
            "value": 0.0018276214832440019,
            "unit": "s",
            "range": 0.0007221624837256968
          },
          {
            "name": "concatenate_N1_GPU - POWER",
            "value": 58.89474846391289,
            "unit": "W",
            "range": 3.587865781955533
          },
          {
            "name": "concatenate_N1_GPU - CPU_UTIL",
            "value": 45.14118352965486,
            "unit": "%",
            "range": 7.4963856414146255
          },
          {
            "name": "concatenate_N1_GPU - GPU_UTIL",
            "value": 8.69380578994751,
            "unit": "%",
            "range": 0.6783586640173771
          },
          {
            "name": "heat_benchmarks_N4_CPU - RUNTIME",
            "value": 23.54559326171875,
            "unit": "s",
            "range": 3.524371862411499
          },
          {
            "name": "heat_benchmarks_N4_CPU - ENERGY",
            "value": 1.3700408662248047,
            "unit": "kJ",
            "range": 0.168573578872834
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.7488749623298645,
            "unit": "s",
            "range": 0.20637628436088562
          },
          {
            "name": "matmul_split_0_N4_CPU - POWER",
            "value": 11.740832030737177,
            "unit": "W",
            "range": 10.080968280851753
          },
          {
            "name": "matmul_split_0_N4_CPU - CPU_UTIL",
            "value": 77.36875597787548,
            "unit": "%",
            "range": 6.620418843915888
          },
          {
            "name": "matmul_split_0_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.7507434487342834,
            "unit": "s",
            "range": 0.23809711635112762
          },
          {
            "name": "matmul_split_1_N4_CPU - POWER",
            "value": 10.343256017021238,
            "unit": "W",
            "range": 9.027969603912076
          },
          {
            "name": "matmul_split_1_N4_CPU - CPU_UTIL",
            "value": 74.99320659246052,
            "unit": "%",
            "range": 9.952155064753551
          },
          {
            "name": "matmul_split_1_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "qr_split_0_N4_CPU - RUNTIME",
            "value": 4.167557716369629,
            "unit": "s",
            "range": 0.985819399356842
          },
          {
            "name": "qr_split_0_N4_CPU - POWER",
            "value": 47.33867938977615,
            "unit": "W",
            "range": 4.823563446959582
          },
          {
            "name": "qr_split_0_N4_CPU - CPU_UTIL",
            "value": 73.63816512683445,
            "unit": "%",
            "range": 9.589116274137623
          },
          {
            "name": "qr_split_0_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "qr_split_1_N4_CPU - RUNTIME",
            "value": 3.9527676105499268,
            "unit": "s",
            "range": 0.615345299243927
          },
          {
            "name": "qr_split_1_N4_CPU - POWER",
            "value": 41.77206932959007,
            "unit": "W",
            "range": 5.166125759971525
          },
          {
            "name": "qr_split_1_N4_CPU - CPU_UTIL",
            "value": 73.95822614200958,
            "unit": "%",
            "range": 8.881172420680425
          },
          {
            "name": "qr_split_1_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 1.7358452081680298,
            "unit": "s",
            "range": 0.24294890463352203
          },
          {
            "name": "lanczos_N4_CPU - POWER",
            "value": 25.46423167363877,
            "unit": "W",
            "range": 12.131512162573404
          },
          {
            "name": "lanczos_N4_CPU - CPU_UTIL",
            "value": 74.40993690044817,
            "unit": "%",
            "range": 8.869536664543764
          },
          {
            "name": "lanczos_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.23207037150859833,
            "unit": "s",
            "range": 0.03904750943183899
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - POWER",
            "value": 7.399255819553825,
            "unit": "W",
            "range": 0.14945703216279413
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - CPU_UTIL",
            "value": 74.42710148699902,
            "unit": "%",
            "range": 8.894009317106217
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.27470317482948303,
            "unit": "s",
            "range": 0.03593892604112625
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - POWER",
            "value": 7.394127602994051,
            "unit": "W",
            "range": 0.15545520530030787
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - CPU_UTIL",
            "value": 74.50863575841542,
            "unit": "%",
            "range": 8.916957025389392
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 2.6261539459228516,
            "unit": "s",
            "range": 0.1911013275384903
          },
          {
            "name": "kmeans_N4_CPU - POWER",
            "value": 34.97328912902122,
            "unit": "W",
            "range": 9.294320053573289
          },
          {
            "name": "kmeans_N4_CPU - CPU_UTIL",
            "value": 73.43969771355258,
            "unit": "%",
            "range": 9.487498623794925
          },
          {
            "name": "kmeans_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 2.696291446685791,
            "unit": "s",
            "range": 0.36521947383880615
          },
          {
            "name": "kmedians_N4_CPU - POWER",
            "value": 38.160875599547836,
            "unit": "W",
            "range": 8.504131178949317
          },
          {
            "name": "kmedians_N4_CPU - CPU_UTIL",
            "value": 73.2038280549415,
            "unit": "%",
            "range": 10.14531740837955
          },
          {
            "name": "kmedians_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 2.854750156402588,
            "unit": "s",
            "range": 0.6489273905754089
          },
          {
            "name": "kmedoids_N4_CPU - POWER",
            "value": 38.85079137119827,
            "unit": "W",
            "range": 13.233584434205907
          },
          {
            "name": "kmedoids_N4_CPU - CPU_UTIL",
            "value": 73.09011542019375,
            "unit": "%",
            "range": 9.822189008817647
          },
          {
            "name": "kmedoids_N4_CPU - GPU_UTIL",
            "value": 0.7073125213384628,
            "unit": "%",
            "range": 0.011647036671638489
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 0.8948338627815247,
            "unit": "s",
            "range": 0.2226608246564865
          },
          {
            "name": "reshape_N4_CPU - POWER",
            "value": 13.87136948307544,
            "unit": "W",
            "range": 12.597901122002298
          },
          {
            "name": "reshape_N4_CPU - CPU_UTIL",
            "value": 72.64818001220827,
            "unit": "%",
            "range": 10.130456939219796
          },
          {
            "name": "reshape_N4_CPU - GPU_UTIL",
            "value": 0.7517957359552383,
            "unit": "%",
            "range": 0.14509668052196503
          },
          {
            "name": "concatenate_N4_CPU - RUNTIME",
            "value": 1.0283732414245605,
            "unit": "s",
            "range": 0.2070590704679489
          },
          {
            "name": "concatenate_N4_CPU - POWER",
            "value": 33.440257419671475,
            "unit": "W",
            "range": 16.56492281485241
          },
          {
            "name": "concatenate_N4_CPU - CPU_UTIL",
            "value": 73.66604300993778,
            "unit": "%",
            "range": 9.302388290196156
          },
          {
            "name": "concatenate_N4_CPU - GPU_UTIL",
            "value": 0.7162292927503586,
            "unit": "%",
            "range": 0.038397350907325746
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Claudia Comito",
            "username": "ClaudiaComito",
            "email": "39374113+ClaudiaComito@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ece385a61714713bf1f51553019eac3ea64d85d8",
          "message": "add SECURITY.md and vulnerability report template (#1221)",
          "timestamp": "2023-09-26T16:00:40Z",
          "url": "https://github.com/helmholtz-analytics/heat/commit/ece385a61714713bf1f51553019eac3ea64d85d8"
        },
        "date": 1696508303354,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "heat_benchmarks_N1_GPU - RUNTIME",
            "value": 56.8642578125,
            "unit": "s",
            "range": 6.704440593719482
          },
          {
            "name": "heat_benchmarks_N1_GPU - ENERGY",
            "value": 5.109793289992188,
            "unit": "kJ",
            "range": 0.8197911708591947
          },
          {
            "name": "matmul_split_0_N1_GPU - RUNTIME",
            "value": 0.0026460334192961454,
            "unit": "s",
            "range": 0.006360592320561409
          },
          {
            "name": "matmul_split_0_N1_GPU - POWER",
            "value": 45.76083094424498,
            "unit": "W",
            "range": 6.692159139071216
          },
          {
            "name": "matmul_split_0_N1_GPU - CPU_UTIL",
            "value": 28.252526838246474,
            "unit": "%",
            "range": 2.8841794673902132
          },
          {
            "name": "matmul_split_0_N1_GPU - GPU_UTIL",
            "value": 12.270692992210389,
            "unit": "%",
            "range": 10.07175292360627
          },
          {
            "name": "matmul_split_1_N1_GPU - RUNTIME",
            "value": 0.0002795342297758907,
            "unit": "s",
            "range": 0.00010195645882049575
          },
          {
            "name": "matmul_split_1_N1_GPU - POWER",
            "value": 46.02612569096517,
            "unit": "W",
            "range": 6.1707290059658035
          },
          {
            "name": "matmul_split_1_N1_GPU - CPU_UTIL",
            "value": 28.36609011746439,
            "unit": "%",
            "range": 2.9381365154561205
          },
          {
            "name": "matmul_split_1_N1_GPU - GPU_UTIL",
            "value": 12.277820229530334,
            "unit": "%",
            "range": 10.056639841166337
          },
          {
            "name": "qr_split_0_N1_GPU - RUNTIME",
            "value": 0.3277803957462311,
            "unit": "s",
            "range": 0.08786257356405258
          },
          {
            "name": "qr_split_0_N1_GPU - POWER",
            "value": 48.081154516841345,
            "unit": "W",
            "range": 2.8994502672083966
          },
          {
            "name": "qr_split_0_N1_GPU - CPU_UTIL",
            "value": 29.004810855408806,
            "unit": "%",
            "range": 3.2999701399194663
          },
          {
            "name": "qr_split_0_N1_GPU - GPU_UTIL",
            "value": 12.342391467094421,
            "unit": "%",
            "range": 9.94245797585625
          },
          {
            "name": "qr_split_1_N1_GPU - RUNTIME",
            "value": 0.27536240220069885,
            "unit": "s",
            "range": 0.07182753831148148
          },
          {
            "name": "qr_split_1_N1_GPU - POWER",
            "value": 50.88762477741142,
            "unit": "W",
            "range": 6.601916312990035
          },
          {
            "name": "qr_split_1_N1_GPU - CPU_UTIL",
            "value": 30.068108957413482,
            "unit": "%",
            "range": 4.203294795483068
          },
          {
            "name": "qr_split_1_N1_GPU - GPU_UTIL",
            "value": 12.422252750396728,
            "unit": "%",
            "range": 9.788724743433503
          },
          {
            "name": "lanczos_N1_GPU - RUNTIME",
            "value": 0.6537602543830872,
            "unit": "s",
            "range": 0.1355219930410385
          },
          {
            "name": "lanczos_N1_GPU - POWER",
            "value": 51.982176734785,
            "unit": "W",
            "range": 7.112145258986832
          },
          {
            "name": "lanczos_N1_GPU - CPU_UTIL",
            "value": 31.499746119870203,
            "unit": "%",
            "range": 5.671968196170039
          },
          {
            "name": "lanczos_N1_GPU - GPU_UTIL",
            "value": 11.990169882774353,
            "unit": "%",
            "range": 8.471940646241647
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - RUNTIME",
            "value": 0.10101678222417831,
            "unit": "s",
            "range": 0.04324260354042053
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - POWER",
            "value": 52.21384349429161,
            "unit": "W",
            "range": 6.537984843017739
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - CPU_UTIL",
            "value": 32.18058183944752,
            "unit": "%",
            "range": 6.280408965880001
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - GPU_UTIL",
            "value": 10.954746127128601,
            "unit": "%",
            "range": 5.530181301548397
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - RUNTIME",
            "value": 0.1673455536365509,
            "unit": "s",
            "range": 0.028592094779014587
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - POWER",
            "value": 52.163969468532244,
            "unit": "W",
            "range": 6.14374860160727
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - CPU_UTIL",
            "value": 32.316447032846526,
            "unit": "%",
            "range": 6.40367290980946
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - GPU_UTIL",
            "value": 10.602184629440307,
            "unit": "%",
            "range": 4.580064871817882
          },
          {
            "name": "kmeans_N1_GPU - RUNTIME",
            "value": 9.477167129516602,
            "unit": "s",
            "range": 1.4648202657699585
          },
          {
            "name": "kmeans_N1_GPU - POWER",
            "value": 83.94590095497622,
            "unit": "W",
            "range": 6.221202055307311
          },
          {
            "name": "kmeans_N1_GPU - CPU_UTIL",
            "value": 33.25711607088372,
            "unit": "%",
            "range": 7.581134399749419
          },
          {
            "name": "kmeans_N1_GPU - GPU_UTIL",
            "value": 9.303547286987305,
            "unit": "%",
            "range": 2.244576603159259
          },
          {
            "name": "kmedians_N1_GPU - RUNTIME",
            "value": 21.409753799438477,
            "unit": "s",
            "range": 3.329479217529297
          },
          {
            "name": "kmedians_N1_GPU - POWER",
            "value": 88.36187534257554,
            "unit": "W",
            "range": 7.349619374359134
          },
          {
            "name": "kmedians_N1_GPU - CPU_UTIL",
            "value": 36.40648456487464,
            "unit": "%",
            "range": 11.662898101312177
          },
          {
            "name": "kmedians_N1_GPU - GPU_UTIL",
            "value": 11.21908483505249,
            "unit": "%",
            "range": 6.6391865710752676
          },
          {
            "name": "kmedoids_N1_GPU - RUNTIME",
            "value": 23.780385971069336,
            "unit": "s",
            "range": 3.225306749343872
          },
          {
            "name": "kmedoids_N1_GPU - POWER",
            "value": 88.54284005933417,
            "unit": "W",
            "range": 7.8482009926314085
          },
          {
            "name": "kmedoids_N1_GPU - CPU_UTIL",
            "value": 37.74755620104101,
            "unit": "%",
            "range": 13.061572741592482
          },
          {
            "name": "kmedoids_N1_GPU - GPU_UTIL",
            "value": 12.65042142868042,
            "unit": "%",
            "range": 9.801123579122358
          },
          {
            "name": "reshape_N1_GPU - RUNTIME",
            "value": 0.0004032134893350303,
            "unit": "s",
            "range": 0.00016764758038334548
          },
          {
            "name": "reshape_N1_GPU - POWER",
            "value": 59.414530268897764,
            "unit": "W",
            "range": 4.736757725542473
          },
          {
            "name": "reshape_N1_GPU - CPU_UTIL",
            "value": 36.77959055703517,
            "unit": "%",
            "range": 17.827760805752565
          },
          {
            "name": "reshape_N1_GPU - GPU_UTIL",
            "value": 13.28124713897705,
            "unit": "%",
            "range": 9.434378107457409
          },
          {
            "name": "concatenate_N1_GPU - RUNTIME",
            "value": 0.0014678954612463713,
            "unit": "s",
            "range": 0.0005946935852989554
          },
          {
            "name": "concatenate_N1_GPU - POWER",
            "value": 59.456183304314926,
            "unit": "W",
            "range": 4.765116018667447
          },
          {
            "name": "concatenate_N1_GPU - CPU_UTIL",
            "value": 36.77414831578696,
            "unit": "%",
            "range": 17.83089516480273
          },
          {
            "name": "concatenate_N1_GPU - GPU_UTIL",
            "value": 13.282094287872315,
            "unit": "%",
            "range": 9.433975938532377
          },
          {
            "name": "heat_benchmarks_N4_CPU - RUNTIME",
            "value": 18.08098030090332,
            "unit": "s",
            "range": 1.9536446332931519
          },
          {
            "name": "heat_benchmarks_N4_CPU - ENERGY",
            "value": 1.159698593904395,
            "unit": "kJ",
            "range": 0.2789200123377995
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.6571148633956909,
            "unit": "s",
            "range": 0.1328810155391693
          },
          {
            "name": "matmul_split_0_N4_CPU - POWER",
            "value": 9.110737817733057,
            "unit": "W",
            "range": 5.506480869740455
          },
          {
            "name": "matmul_split_0_N4_CPU - CPU_UTIL",
            "value": 84.71423300355269,
            "unit": "%",
            "range": 13.426633677303094
          },
          {
            "name": "matmul_split_0_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.6150993704795837,
            "unit": "s",
            "range": 0.09445267915725708
          },
          {
            "name": "matmul_split_1_N4_CPU - POWER",
            "value": 7.251800516881,
            "unit": "W",
            "range": 0.17525692005690113
          },
          {
            "name": "matmul_split_1_N4_CPU - CPU_UTIL",
            "value": 77.11445903019731,
            "unit": "%",
            "range": 20.73206657779303
          },
          {
            "name": "matmul_split_1_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "qr_split_0_N4_CPU - RUNTIME",
            "value": 3.3225960731506348,
            "unit": "s",
            "range": 0.2991132438182831
          },
          {
            "name": "qr_split_0_N4_CPU - POWER",
            "value": 46.6116600170091,
            "unit": "W",
            "range": 8.839592309167683
          },
          {
            "name": "qr_split_0_N4_CPU - CPU_UTIL",
            "value": 76.50595149391674,
            "unit": "%",
            "range": 20.5503486322189
          },
          {
            "name": "qr_split_0_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "qr_split_1_N4_CPU - RUNTIME",
            "value": 3.2006702423095703,
            "unit": "s",
            "range": 0.3145384192466736
          },
          {
            "name": "qr_split_1_N4_CPU - POWER",
            "value": 40.96622185259399,
            "unit": "W",
            "range": 5.1186242653720795
          },
          {
            "name": "qr_split_1_N4_CPU - CPU_UTIL",
            "value": 75.19291134875493,
            "unit": "%",
            "range": 20.84369330178813
          },
          {
            "name": "qr_split_1_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 1.3180084228515625,
            "unit": "s",
            "range": 0.31274110078811646
          },
          {
            "name": "lanczos_N4_CPU - POWER",
            "value": 26.86804421726236,
            "unit": "W",
            "range": 22.16593473392376
          },
          {
            "name": "lanczos_N4_CPU - CPU_UTIL",
            "value": 74.72568254612531,
            "unit": "%",
            "range": 21.925637411950397
          },
          {
            "name": "lanczos_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.22320988774299622,
            "unit": "s",
            "range": 0.0387376993894577
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - POWER",
            "value": 7.265007160218556,
            "unit": "W",
            "range": 0.15747833965033675
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - CPU_UTIL",
            "value": 74.84948010220221,
            "unit": "%",
            "range": 22.00428703335526
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.25486406683921814,
            "unit": "s",
            "range": 0.05069941282272339
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - POWER",
            "value": 7.253409510008915,
            "unit": "W",
            "range": 0.17212932290106994
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - CPU_UTIL",
            "value": 74.89251044054572,
            "unit": "%",
            "range": 22.019549860730415
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.8114229440689087,
            "unit": "s",
            "range": 0.4608151912689209
          },
          {
            "name": "kmeans_N4_CPU - POWER",
            "value": 33.40270338485798,
            "unit": "W",
            "range": 15.931515168678617
          },
          {
            "name": "kmeans_N4_CPU - CPU_UTIL",
            "value": 74.36685519790915,
            "unit": "%",
            "range": 22.31541089284109
          },
          {
            "name": "kmeans_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 1.870810866355896,
            "unit": "s",
            "range": 0.19969730079174042
          },
          {
            "name": "kmedians_N4_CPU - POWER",
            "value": 39.3697768956486,
            "unit": "W",
            "range": 7.143046764136199
          },
          {
            "name": "kmedians_N4_CPU - CPU_UTIL",
            "value": 72.6715017667345,
            "unit": "%",
            "range": 23.69569113413927
          },
          {
            "name": "kmedians_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 1.953009843826294,
            "unit": "s",
            "range": 0.25879570841789246
          },
          {
            "name": "kmedoids_N4_CPU - POWER",
            "value": 26.605253702138704,
            "unit": "W",
            "range": 14.619276859632588
          },
          {
            "name": "kmedoids_N4_CPU - CPU_UTIL",
            "value": 72.1656347577172,
            "unit": "%",
            "range": 23.619257020449474
          },
          {
            "name": "kmedoids_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 0.7416497468948364,
            "unit": "s",
            "range": 0.14900584518909454
          },
          {
            "name": "reshape_N4_CPU - POWER",
            "value": 7.269806457758621,
            "unit": "W",
            "range": 0.1002591585369334
          },
          {
            "name": "reshape_N4_CPU - CPU_UTIL",
            "value": 72.11577948169882,
            "unit": "%",
            "range": 23.403497702597658
          },
          {
            "name": "reshape_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          },
          {
            "name": "concatenate_N4_CPU - RUNTIME",
            "value": 0.8123544454574585,
            "unit": "s",
            "range": 0.09851332753896713
          },
          {
            "name": "concatenate_N4_CPU - POWER",
            "value": 19.55136037515764,
            "unit": "W",
            "range": 13.946380751791708
          },
          {
            "name": "concatenate_N4_CPU - CPU_UTIL",
            "value": 71.84932970451169,
            "unit": "%",
            "range": 22.41627985942541
          },
          {
            "name": "concatenate_N4_CPU - GPU_UTIL",
            "value": 0.70343017578125,
            "unit": "%",
            "range": 0
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "JuanPedroGHM",
            "username": "JuanPedroGHM",
            "email": "juanpedroghm@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "724a80b256ef0cea9f71dbe12c124d6713bcbb32",
          "message": "Benchmarks: missing branch argument in gitlab pipeline trigger (#1227)\n\n* missing variable in ci call\r\n\r\n* filter status creation",
          "timestamp": "2023-10-09T09:12:06Z",
          "url": "https://github.com/helmholtz-analytics/heat/commit/724a80b256ef0cea9f71dbe12c124d6713bcbb32"
        },
        "date": 1696845330472,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "heat_benchmarks_N1_GPU - RUNTIME",
            "value": 55.78736114501953,
            "unit": "s",
            "range": 11.314675331115723
          },
          {
            "name": "heat_benchmarks_N1_GPU - ENERGY",
            "value": 5.9921207205625,
            "unit": "kJ",
            "range": 0.9108129651380957
          },
          {
            "name": "matmul_split_0_N1_GPU - RUNTIME",
            "value": 0.00757700065150857,
            "unit": "s",
            "range": 0.020734060555696487
          },
          {
            "name": "matmul_split_0_N1_GPU - POWER",
            "value": 49.87639845560975,
            "unit": "W",
            "range": 4.2578509669972915
          },
          {
            "name": "matmul_split_0_N1_GPU - CPU_UTIL",
            "value": 80.82613504594491,
            "unit": "%",
            "range": 4.187355819467698
          },
          {
            "name": "matmul_split_0_N1_GPU - GPU_UTIL",
            "value": 13.687237858772278,
            "unit": "%",
            "range": 9.481088586806862
          },
          {
            "name": "matmul_split_1_N1_GPU - RUNTIME",
            "value": 0.0008374780300073326,
            "unit": "s",
            "range": 0.0012782190460711718
          },
          {
            "name": "matmul_split_1_N1_GPU - POWER",
            "value": 49.96990763858559,
            "unit": "W",
            "range": 4.237705428150179
          },
          {
            "name": "matmul_split_1_N1_GPU - CPU_UTIL",
            "value": 80.72426516180518,
            "unit": "%",
            "range": 4.271062121931783
          },
          {
            "name": "matmul_split_1_N1_GPU - GPU_UTIL",
            "value": 13.707086634635925,
            "unit": "%",
            "range": 9.46321568183581
          },
          {
            "name": "qr_split_0_N1_GPU - RUNTIME",
            "value": 0.35141587257385254,
            "unit": "s",
            "range": 0.10598938167095184
          },
          {
            "name": "qr_split_0_N1_GPU - POWER",
            "value": 50.49995042809904,
            "unit": "W",
            "range": 4.271981589367044
          },
          {
            "name": "qr_split_0_N1_GPU - CPU_UTIL",
            "value": 79.44037041494668,
            "unit": "%",
            "range": 7.349940182904267
          },
          {
            "name": "qr_split_0_N1_GPU - GPU_UTIL",
            "value": 13.798886156082153,
            "unit": "%",
            "range": 9.382926845513005
          },
          {
            "name": "qr_split_1_N1_GPU - RUNTIME",
            "value": 0.298775851726532,
            "unit": "s",
            "range": 0.09018929302692413
          },
          {
            "name": "qr_split_1_N1_GPU - POWER",
            "value": 51.2663354126909,
            "unit": "W",
            "range": 4.4395604434091895
          },
          {
            "name": "qr_split_1_N1_GPU - CPU_UTIL",
            "value": 77.45417177225241,
            "unit": "%",
            "range": 14.40430616344331
          },
          {
            "name": "qr_split_1_N1_GPU - GPU_UTIL",
            "value": 13.898281383514405,
            "unit": "%",
            "range": 9.307776404210498
          },
          {
            "name": "lanczos_N1_GPU - RUNTIME",
            "value": 0.7449082136154175,
            "unit": "s",
            "range": 0.2581082880496979
          },
          {
            "name": "lanczos_N1_GPU - POWER",
            "value": 52.24650799407776,
            "unit": "W",
            "range": 4.726459544135903
          },
          {
            "name": "lanczos_N1_GPU - CPU_UTIL",
            "value": 76.52685264190418,
            "unit": "%",
            "range": 19.860156857683666
          },
          {
            "name": "lanczos_N1_GPU - GPU_UTIL",
            "value": 13.949684476852417,
            "unit": "%",
            "range": 9.300585128714143
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - RUNTIME",
            "value": 0.09399120509624481,
            "unit": "s",
            "range": 0.03459111228585243
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - POWER",
            "value": 53.47623048043302,
            "unit": "W",
            "range": 6.293197403594754
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - CPU_UTIL",
            "value": 76.23630938081583,
            "unit": "%",
            "range": 21.880743008218126
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - GPU_UTIL",
            "value": 13.968425416946411,
            "unit": "%",
            "range": 9.298773155840514
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - RUNTIME",
            "value": 0.1661851406097412,
            "unit": "s",
            "range": 0.02613021433353424
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - POWER",
            "value": 53.57475209775832,
            "unit": "W",
            "range": 6.260267520167685
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - CPU_UTIL",
            "value": 76.17503481439975,
            "unit": "%",
            "range": 22.034403968023184
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - GPU_UTIL",
            "value": 13.968644714355468,
            "unit": "%",
            "range": 9.29856158630112
          },
          {
            "name": "kmeans_N1_GPU - RUNTIME",
            "value": 8.447792053222656,
            "unit": "s",
            "range": 2.281770944595337
          },
          {
            "name": "kmeans_N1_GPU - POWER",
            "value": 101.91252799227972,
            "unit": "W",
            "range": 8.937666323258084
          },
          {
            "name": "kmeans_N1_GPU - CPU_UTIL",
            "value": 75.0273577223065,
            "unit": "%",
            "range": 22.967617013001423
          },
          {
            "name": "kmeans_N1_GPU - GPU_UTIL",
            "value": 14.008415031433106,
            "unit": "%",
            "range": 9.318428734547945
          },
          {
            "name": "kmedians_N1_GPU - RUNTIME",
            "value": 22.354755401611328,
            "unit": "s",
            "range": 4.088697910308838
          },
          {
            "name": "kmedians_N1_GPU - POWER",
            "value": 106.80628565584239,
            "unit": "W",
            "range": 9.96728010439786
          },
          {
            "name": "kmedians_N1_GPU - CPU_UTIL",
            "value": 73.6908958879318,
            "unit": "%",
            "range": 23.217507212690194
          },
          {
            "name": "kmedians_N1_GPU - GPU_UTIL",
            "value": 14.357549285888672,
            "unit": "%",
            "range": 9.29588105167657
          },
          {
            "name": "kmedoids_N1_GPU - RUNTIME",
            "value": 22.532583236694336,
            "unit": "s",
            "range": 4.276721954345703
          },
          {
            "name": "kmedoids_N1_GPU - POWER",
            "value": 106.72286984612796,
            "unit": "W",
            "range": 8.998570050634074
          },
          {
            "name": "kmedoids_N1_GPU - CPU_UTIL",
            "value": 73.08467665285026,
            "unit": "%",
            "range": 23.398685801640372
          },
          {
            "name": "kmedoids_N1_GPU - GPU_UTIL",
            "value": 12.032056713104248,
            "unit": "%",
            "range": 3.237944960632083
          },
          {
            "name": "reshape_N1_GPU - RUNTIME",
            "value": 0.0009456634288653731,
            "unit": "s",
            "range": 0.001419209293089807
          },
          {
            "name": "reshape_N1_GPU - POWER",
            "value": 64.07727708917629,
            "unit": "W",
            "range": 5.308973604782024
          },
          {
            "name": "reshape_N1_GPU - CPU_UTIL",
            "value": 75.42270296280276,
            "unit": "%",
            "range": 23.143238080723744
          },
          {
            "name": "reshape_N1_GPU - GPU_UTIL",
            "value": 14.410399436950684,
            "unit": "%",
            "range": 9.000227251227999
          },
          {
            "name": "concatenate_N1_GPU - RUNTIME",
            "value": 0.0020614624954760075,
            "unit": "s",
            "range": 0.0016353533137589693
          },
          {
            "name": "concatenate_N1_GPU - POWER",
            "value": 64.18778243452019,
            "unit": "W",
            "range": 5.338695737082786
          },
          {
            "name": "concatenate_N1_GPU - CPU_UTIL",
            "value": 75.50128931074994,
            "unit": "%",
            "range": 23.19131588468861
          },
          {
            "name": "concatenate_N1_GPU - GPU_UTIL",
            "value": 14.419466686248779,
            "unit": "%",
            "range": 8.993723661837437
          },
          {
            "name": "heat_benchmarks_N4_CPU - RUNTIME",
            "value": 16.911808013916016,
            "unit": "s",
            "range": 1.8824572563171387
          },
          {
            "name": "heat_benchmarks_N4_CPU - ENERGY",
            "value": 1.3133995955104978,
            "unit": "kJ",
            "range": 0.25536059514139076
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.6010230183601379,
            "unit": "s",
            "range": 0.11481460183858871
          },
          {
            "name": "matmul_split_0_N4_CPU - POWER",
            "value": 11.38115920202092,
            "unit": "W",
            "range": 10.623808934870924
          },
          {
            "name": "matmul_split_0_N4_CPU - CPU_UTIL",
            "value": 87.21788016559738,
            "unit": "%",
            "range": 1.513188864337606
          },
          {
            "name": "matmul_split_0_N4_CPU - GPU_UTIL",
            "value": 1.27960205078125,
            "unit": "%",
            "range": 0.2880859375
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.5360350608825684,
            "unit": "s",
            "range": 0.03402045741677284
          },
          {
            "name": "matmul_split_1_N4_CPU - POWER",
            "value": 11.255847421647635,
            "unit": "W",
            "range": 10.695264823780148
          },
          {
            "name": "matmul_split_1_N4_CPU - CPU_UTIL",
            "value": 87.96077910946954,
            "unit": "%",
            "range": 2.5488107656409307
          },
          {
            "name": "matmul_split_1_N4_CPU - GPU_UTIL",
            "value": 1.27960205078125,
            "unit": "%",
            "range": 0.2880859375
          },
          {
            "name": "qr_split_0_N4_CPU - RUNTIME",
            "value": 3.1217117309570312,
            "unit": "s",
            "range": 0.23657777905464172
          },
          {
            "name": "qr_split_0_N4_CPU - POWER",
            "value": 53.824509735915434,
            "unit": "W",
            "range": 11.091346279890912
          },
          {
            "name": "qr_split_0_N4_CPU - CPU_UTIL",
            "value": 88.30191019159113,
            "unit": "%",
            "range": 2.59393527111396
          },
          {
            "name": "qr_split_0_N4_CPU - GPU_UTIL",
            "value": 1.2905536726117135,
            "unit": "%",
            "range": 0.2673067836831179
          },
          {
            "name": "qr_split_1_N4_CPU - RUNTIME",
            "value": 2.9718658924102783,
            "unit": "s",
            "range": 0.13829568028450012
          },
          {
            "name": "qr_split_1_N4_CPU - POWER",
            "value": 52.587519540333666,
            "unit": "W",
            "range": 15.179178968834764
          },
          {
            "name": "qr_split_1_N4_CPU - CPU_UTIL",
            "value": 87.65507909479079,
            "unit": "%",
            "range": 1.7792866572449655
          },
          {
            "name": "qr_split_1_N4_CPU - GPU_UTIL",
            "value": 1.4155546963214873,
            "unit": "%",
            "range": 0.1911351522088796
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 1.1821396350860596,
            "unit": "s",
            "range": 0.1332835704088211
          },
          {
            "name": "lanczos_N4_CPU - POWER",
            "value": 18.78812961354826,
            "unit": "W",
            "range": 15.972556799972525
          },
          {
            "name": "lanczos_N4_CPU - CPU_UTIL",
            "value": 87.66298268581231,
            "unit": "%",
            "range": 1.87332188637324
          },
          {
            "name": "lanczos_N4_CPU - GPU_UTIL",
            "value": 1.47003173828125,
            "unit": "%",
            "range": 0.1695679233807162
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.19667509198188782,
            "unit": "s",
            "range": 0.017168203368782997
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - POWER",
            "value": 20.93181758618398,
            "unit": "W",
            "range": 16.71670200702643
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - CPU_UTIL",
            "value": 87.48903512155793,
            "unit": "%",
            "range": 1.945130223157593
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - GPU_UTIL",
            "value": 1.47003173828125,
            "unit": "%",
            "range": 0.1695679233807162
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.2374892681837082,
            "unit": "s",
            "range": 0.02720005437731743
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - POWER",
            "value": 21.46370873371577,
            "unit": "W",
            "range": 17.254480983785477
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - CPU_UTIL",
            "value": 87.46012876538353,
            "unit": "%",
            "range": 1.9564688638069399
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - GPU_UTIL",
            "value": 1.47003173828125,
            "unit": "%",
            "range": 0.1695679233807162
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.6521316766738892,
            "unit": "s",
            "range": 0.18586091697216034
          },
          {
            "name": "kmeans_N4_CPU - POWER",
            "value": 42.54782403576265,
            "unit": "W",
            "range": 22.301292111283963
          },
          {
            "name": "kmeans_N4_CPU - CPU_UTIL",
            "value": 87.56793984403926,
            "unit": "%",
            "range": 1.9139989751689375
          },
          {
            "name": "kmeans_N4_CPU - GPU_UTIL",
            "value": 1.43402099609375,
            "unit": "%",
            "range": 0.2092010586860411
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 1.9199142456054688,
            "unit": "s",
            "range": 0.18181312084197998
          },
          {
            "name": "kmedians_N4_CPU - POWER",
            "value": 46.96565326160652,
            "unit": "W",
            "range": 25.550195919071577
          },
          {
            "name": "kmedians_N4_CPU - CPU_UTIL",
            "value": 87.99547666908745,
            "unit": "%",
            "range": 2.1844286848960723
          },
          {
            "name": "kmedians_N4_CPU - GPU_UTIL",
            "value": 1.41082763671875,
            "unit": "%",
            "range": 0.3036147820946968
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 1.9111286401748657,
            "unit": "s",
            "range": 0.20731709897518158
          },
          {
            "name": "kmedoids_N4_CPU - POWER",
            "value": 52.885648213738044,
            "unit": "W",
            "range": 18.348981076390626
          },
          {
            "name": "kmedoids_N4_CPU - CPU_UTIL",
            "value": 88.41263267889438,
            "unit": "%",
            "range": 3.014496086022289
          },
          {
            "name": "kmedoids_N4_CPU - GPU_UTIL",
            "value": 1.42364501953125,
            "unit": "%",
            "range": 0.3220898698058779
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 0.6780017018318176,
            "unit": "s",
            "range": 0.10253068059682846
          },
          {
            "name": "reshape_N4_CPU - POWER",
            "value": 15.546392996434914,
            "unit": "W",
            "range": 13.964819009891047
          },
          {
            "name": "reshape_N4_CPU - CPU_UTIL",
            "value": 89.34740073230537,
            "unit": "%",
            "range": 3.754251956116911
          },
          {
            "name": "reshape_N4_CPU - GPU_UTIL",
            "value": 1.42364501953125,
            "unit": "%",
            "range": 0.3220898698058779
          },
          {
            "name": "concatenate_N4_CPU - RUNTIME",
            "value": 0.6788491010665894,
            "unit": "s",
            "range": 0.05435426905751228
          },
          {
            "name": "concatenate_N4_CPU - POWER",
            "value": 21.21662812134195,
            "unit": "W",
            "range": 12.703090433712195
          },
          {
            "name": "concatenate_N4_CPU - CPU_UTIL",
            "value": 89.50606118118604,
            "unit": "%",
            "range": 3.953018628687951
          },
          {
            "name": "concatenate_N4_CPU - GPU_UTIL",
            "value": 1.42364501953125,
            "unit": "%",
            "range": 0.3220898698058779
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Sai-Suraj-27",
            "username": "Sai-Suraj-27",
            "email": "sai.suraj.27.729@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a32efdb1df74d0b1c57931d6f5e441d1ebff09b9",
          "message": "Updated `.pre-commit-config.yaml` file and reformatted few files for better readability. (#1211)\n\n* Used pylint automated tool to update from .format to f-strings.\r\n\r\n* Updated the pre-commit configuration file.\r\n\r\n* Reformatted 2 more files to use f-strings.\r\n\r\n* Added new check that verifies .toml files in the repo, and updated black repo link.\r\n\r\n---------\r\n\r\nCo-authored-by: Claudia Comito <39374113+ClaudiaComito@users.noreply.github.com>",
          "timestamp": "2023-10-12T04:37:21Z",
          "url": "https://github.com/helmholtz-analytics/heat/commit/a32efdb1df74d0b1c57931d6f5e441d1ebff09b9"
        },
        "date": 1697086999623,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "heat_benchmarks_N1_GPU - RUNTIME",
            "value": 60.47065353393555,
            "unit": "s",
            "range": 8.831391334533691
          },
          {
            "name": "heat_benchmarks_N1_GPU - ENERGY",
            "value": 5.563966734054688,
            "unit": "kJ",
            "range": 0.7654355512640935
          },
          {
            "name": "matmul_split_0_N1_GPU - RUNTIME",
            "value": 0.005564832128584385,
            "unit": "s",
            "range": 0.015212520956993103
          },
          {
            "name": "matmul_split_0_N1_GPU - POWER",
            "value": 51.508849404232535,
            "unit": "W",
            "range": 5.73496410292711
          },
          {
            "name": "matmul_split_0_N1_GPU - CPU_UTIL",
            "value": 24.1721173698035,
            "unit": "%",
            "range": 3.1628969431341782
          },
          {
            "name": "matmul_split_0_N1_GPU - GPU_UTIL",
            "value": 14.353278493881225,
            "unit": "%",
            "range": 9.265653891818896
          },
          {
            "name": "matmul_split_1_N1_GPU - RUNTIME",
            "value": 0.00029110832838341594,
            "unit": "s",
            "range": 0.00012181716010672972
          },
          {
            "name": "matmul_split_1_N1_GPU - POWER",
            "value": 51.71298598240021,
            "unit": "W",
            "range": 5.675946711286795
          },
          {
            "name": "matmul_split_1_N1_GPU - CPU_UTIL",
            "value": 24.290434500686892,
            "unit": "%",
            "range": 3.21988751431144
          },
          {
            "name": "matmul_split_1_N1_GPU - GPU_UTIL",
            "value": 14.366141414642334,
            "unit": "%",
            "range": 9.251776304066254
          },
          {
            "name": "qr_split_0_N1_GPU - RUNTIME",
            "value": 0.36606013774871826,
            "unit": "s",
            "range": 0.13507241010665894
          },
          {
            "name": "qr_split_0_N1_GPU - POWER",
            "value": 52.91281844041241,
            "unit": "W",
            "range": 5.771813800253857
          },
          {
            "name": "qr_split_0_N1_GPU - CPU_UTIL",
            "value": 25.003572445928643,
            "unit": "%",
            "range": 3.5402755909495855
          },
          {
            "name": "qr_split_0_N1_GPU - GPU_UTIL",
            "value": 14.40151505470276,
            "unit": "%",
            "range": 9.215035950511071
          },
          {
            "name": "qr_split_1_N1_GPU - RUNTIME",
            "value": 0.304084837436676,
            "unit": "s",
            "range": 0.13362672924995422
          },
          {
            "name": "qr_split_1_N1_GPU - POWER",
            "value": 54.702601950469514,
            "unit": "W",
            "range": 6.592191763264783
          },
          {
            "name": "qr_split_1_N1_GPU - CPU_UTIL",
            "value": 26.1223368365623,
            "unit": "%",
            "range": 4.0695526843861956
          },
          {
            "name": "qr_split_1_N1_GPU - GPU_UTIL",
            "value": 14.419933652877807,
            "unit": "%",
            "range": 9.197622630705506
          },
          {
            "name": "lanczos_N1_GPU - RUNTIME",
            "value": 0.6866452097892761,
            "unit": "s",
            "range": 0.27626144886016846
          },
          {
            "name": "lanczos_N1_GPU - POWER",
            "value": 55.62813734966812,
            "unit": "W",
            "range": 6.762977754458709
          },
          {
            "name": "lanczos_N1_GPU - CPU_UTIL",
            "value": 27.092713952064663,
            "unit": "%",
            "range": 4.616533756013383
          },
          {
            "name": "lanczos_N1_GPU - GPU_UTIL",
            "value": 14.433641815185547,
            "unit": "%",
            "range": 9.199194859851586
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - RUNTIME",
            "value": 0.10040156543254852,
            "unit": "s",
            "range": 0.04669195041060448
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - POWER",
            "value": 56.481289360716445,
            "unit": "W",
            "range": 8.05378591057715
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - CPU_UTIL",
            "value": 27.533158800419102,
            "unit": "%",
            "range": 4.920966610996647
          },
          {
            "name": "hierachical_svd_rank_N1_GPU - GPU_UTIL",
            "value": 14.446846675872802,
            "unit": "%",
            "range": 9.219068364816176
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - RUNTIME",
            "value": 0.1675301343202591,
            "unit": "s",
            "range": 0.037118099629879
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - POWER",
            "value": 56.74335315580638,
            "unit": "W",
            "range": 8.218231770131421
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - CPU_UTIL",
            "value": 27.623730917864407,
            "unit": "%",
            "range": 4.997210671930903
          },
          {
            "name": "hierachical_svd_tol_N1_GPU - GPU_UTIL",
            "value": 14.449577379226685,
            "unit": "%",
            "range": 9.225899762252952
          },
          {
            "name": "kmeans_N1_GPU - RUNTIME",
            "value": 9.90272045135498,
            "unit": "s",
            "range": 1.9667556285858154
          },
          {
            "name": "kmeans_N1_GPU - POWER",
            "value": 88.64328459515215,
            "unit": "W",
            "range": 6.273595051921986
          },
          {
            "name": "kmeans_N1_GPU - CPU_UTIL",
            "value": 28.089565961523135,
            "unit": "%",
            "range": 4.467811764602459
          },
          {
            "name": "kmeans_N1_GPU - GPU_UTIL",
            "value": 14.402999877929688,
            "unit": "%",
            "range": 9.317269100906994
          },
          {
            "name": "kmedians_N1_GPU - RUNTIME",
            "value": 24.261812210083008,
            "unit": "s",
            "range": 3.9632585048675537
          },
          {
            "name": "kmedians_N1_GPU - POWER",
            "value": 90.40016777094192,
            "unit": "W",
            "range": 7.223480524141712
          },
          {
            "name": "kmedians_N1_GPU - CPU_UTIL",
            "value": 28.43067134765505,
            "unit": "%",
            "range": 4.342067047643432
          },
          {
            "name": "kmedians_N1_GPU - GPU_UTIL",
            "value": 14.103629207611084,
            "unit": "%",
            "range": 8.345046349638652
          },
          {
            "name": "kmedoids_N1_GPU - RUNTIME",
            "value": 24.04513931274414,
            "unit": "s",
            "range": 4.235551834106445
          },
          {
            "name": "kmedoids_N1_GPU - POWER",
            "value": 91.61068009051056,
            "unit": "W",
            "range": 6.537435799636333
          },
          {
            "name": "kmedoids_N1_GPU - CPU_UTIL",
            "value": 29.3411847893787,
            "unit": "%",
            "range": 3.9011727976445263
          },
          {
            "name": "kmedoids_N1_GPU - GPU_UTIL",
            "value": 12.494498682022094,
            "unit": "%",
            "range": 3.807575650234944
          },
          {
            "name": "reshape_N1_GPU - RUNTIME",
            "value": 0.0005474090576171875,
            "unit": "s",
            "range": 0.00020142360881436616
          },
          {
            "name": "reshape_N1_GPU - POWER",
            "value": 64.65211864918746,
            "unit": "W",
            "range": 6.574677712469169
          },
          {
            "name": "reshape_N1_GPU - CPU_UTIL",
            "value": 28.899277628235303,
            "unit": "%",
            "range": 4.178183664618654
          },
          {
            "name": "reshape_N1_GPU - GPU_UTIL",
            "value": 15.410915660858155,
            "unit": "%",
            "range": 8.633014947251194
          },
          {
            "name": "concatenate_N1_GPU - RUNTIME",
            "value": 0.002167129423469305,
            "unit": "s",
            "range": 0.001327261794358492
          },
          {
            "name": "concatenate_N1_GPU - POWER",
            "value": 64.70961432425405,
            "unit": "W",
            "range": 6.569522518539486
          },
          {
            "name": "concatenate_N1_GPU - CPU_UTIL",
            "value": 28.89971835734907,
            "unit": "%",
            "range": 4.178530382178055
          },
          {
            "name": "concatenate_N1_GPU - GPU_UTIL",
            "value": 15.416102600097656,
            "unit": "%",
            "range": 8.6294163339939
          },
          {
            "name": "heat_benchmarks_N4_CPU - RUNTIME",
            "value": 17.53148651123047,
            "unit": "s",
            "range": 1.6468853950500488
          },
          {
            "name": "heat_benchmarks_N4_CPU - ENERGY",
            "value": 967.5927642938475,
            "unit": "J",
            "range": 224.7602839361693
          },
          {
            "name": "matmul_split_0_N4_CPU - RUNTIME",
            "value": 0.6468690633773804,
            "unit": "s",
            "range": 0.14477534592151642
          },
          {
            "name": "matmul_split_0_N4_CPU - POWER",
            "value": 12.571030422055518,
            "unit": "W",
            "range": 10.932127495992125
          },
          {
            "name": "matmul_split_0_N4_CPU - CPU_UTIL",
            "value": 31.40404209271974,
            "unit": "%",
            "range": 0.11607318805195059
          },
          {
            "name": "matmul_split_0_N4_CPU - GPU_UTIL",
            "value": 1.35162353515625,
            "unit": "%",
            "range": 0.216064453125
          },
          {
            "name": "matmul_split_1_N4_CPU - RUNTIME",
            "value": 0.6135107278823853,
            "unit": "s",
            "range": 0.03035014308989048
          },
          {
            "name": "matmul_split_1_N4_CPU - POWER",
            "value": 14.463759330060759,
            "unit": "W",
            "range": 13.528066870198531
          },
          {
            "name": "matmul_split_1_N4_CPU - CPU_UTIL",
            "value": 31.257841524224023,
            "unit": "%",
            "range": 0.12291488170155088
          },
          {
            "name": "matmul_split_1_N4_CPU - GPU_UTIL",
            "value": 1.3426623851060868,
            "unit": "%",
            "range": 0.21474584312521502
          },
          {
            "name": "qr_split_0_N4_CPU - RUNTIME",
            "value": 3.2289490699768066,
            "unit": "s",
            "range": 0.22846408188343048
          },
          {
            "name": "qr_split_0_N4_CPU - POWER",
            "value": 43.620479905060826,
            "unit": "W",
            "range": 13.596174228164813
          },
          {
            "name": "qr_split_0_N4_CPU - CPU_UTIL",
            "value": 31.7698472149623,
            "unit": "%",
            "range": 1.5093889488426258
          },
          {
            "name": "qr_split_0_N4_CPU - GPU_UTIL",
            "value": 1.2875876486301423,
            "unit": "%",
            "range": 0.2726999845454847
          },
          {
            "name": "qr_split_1_N4_CPU - RUNTIME",
            "value": 3.2962403297424316,
            "unit": "s",
            "range": 0.31408044695854187
          },
          {
            "name": "qr_split_1_N4_CPU - POWER",
            "value": 43.98138077980103,
            "unit": "W",
            "range": 11.978885291413313
          },
          {
            "name": "qr_split_1_N4_CPU - CPU_UTIL",
            "value": 32.375693861338064,
            "unit": "%",
            "range": 1.9918572054747112
          },
          {
            "name": "qr_split_1_N4_CPU - GPU_UTIL",
            "value": 1.2879200279712677,
            "unit": "%",
            "range": 0.272086451442363
          },
          {
            "name": "lanczos_N4_CPU - RUNTIME",
            "value": 1.0464246273040771,
            "unit": "s",
            "range": 0.07311129570007324
          },
          {
            "name": "lanczos_N4_CPU - POWER",
            "value": 14.624875787591842,
            "unit": "W",
            "range": 13.231589369538654
          },
          {
            "name": "lanczos_N4_CPU - CPU_UTIL",
            "value": 32.78580696775307,
            "unit": "%",
            "range": 2.390155261761452
          },
          {
            "name": "lanczos_N4_CPU - GPU_UTIL",
            "value": 1.3358707278966904,
            "unit": "%",
            "range": 0.21598177054882042
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - RUNTIME",
            "value": 0.17017874121665955,
            "unit": "s",
            "range": 0.021947436034679413
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - POWER",
            "value": 14.952837480389409,
            "unit": "W",
            "range": 13.992734153366397
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - CPU_UTIL",
            "value": 32.70382876882083,
            "unit": "%",
            "range": 2.1836661542611058
          },
          {
            "name": "hierachical_svd_rank_N4_CPU - GPU_UTIL",
            "value": 1.34307861328125,
            "unit": "%",
            "range": 0.21473274831343772
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - RUNTIME",
            "value": 0.2375410795211792,
            "unit": "s",
            "range": 0.04626565799117088
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - POWER",
            "value": 14.890715708034211,
            "unit": "W",
            "range": 14.00552295537412
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - CPU_UTIL",
            "value": 32.63882381640006,
            "unit": "%",
            "range": 2.048723859712222
          },
          {
            "name": "hierachical_svd_tol_N4_CPU - GPU_UTIL",
            "value": 1.34307861328125,
            "unit": "%",
            "range": 0.21473274831343772
          },
          {
            "name": "kmeans_N4_CPU - RUNTIME",
            "value": 1.668572187423706,
            "unit": "s",
            "range": 0.32434043288230896
          },
          {
            "name": "kmeans_N4_CPU - POWER",
            "value": 29.742742630812813,
            "unit": "W",
            "range": 22.155627683338274
          },
          {
            "name": "kmeans_N4_CPU - CPU_UTIL",
            "value": 32.58798650586897,
            "unit": "%",
            "range": 2.403376559959448
          },
          {
            "name": "kmeans_N4_CPU - GPU_UTIL",
            "value": 1.34307861328125,
            "unit": "%",
            "range": 0.21473274831343772
          },
          {
            "name": "kmedians_N4_CPU - RUNTIME",
            "value": 2.0321249961853027,
            "unit": "s",
            "range": 0.26462602615356445
          },
          {
            "name": "kmedians_N4_CPU - POWER",
            "value": 34.81611245040275,
            "unit": "W",
            "range": 13.123226951801648
          },
          {
            "name": "kmedians_N4_CPU - CPU_UTIL",
            "value": 32.3447531751636,
            "unit": "%",
            "range": 2.2699938816082343
          },
          {
            "name": "kmedians_N4_CPU - GPU_UTIL",
            "value": 1.4093509674072267,
            "unit": "%",
            "range": 0.029263662767613337
          },
          {
            "name": "kmedoids_N4_CPU - RUNTIME",
            "value": 1.726906180381775,
            "unit": "s",
            "range": 0.426612913608551
          },
          {
            "name": "kmedoids_N4_CPU - POWER",
            "value": 32.39918642533284,
            "unit": "W",
            "range": 24.32963848259863
          },
          {
            "name": "kmedoids_N4_CPU - CPU_UTIL",
            "value": 31.918579446000745,
            "unit": "%",
            "range": 1.869685884222718
          },
          {
            "name": "kmedoids_N4_CPU - GPU_UTIL",
            "value": 1.4785373151302337,
            "unit": "%",
            "range": 0.1648080786300417
          },
          {
            "name": "reshape_N4_CPU - RUNTIME",
            "value": 0.741828441619873,
            "unit": "s",
            "range": 0.1595792919397354
          },
          {
            "name": "reshape_N4_CPU - POWER",
            "value": 14.691242601790776,
            "unit": "W",
            "range": 14.160629189615463
          },
          {
            "name": "reshape_N4_CPU - CPU_UTIL",
            "value": 31.9860676779026,
            "unit": "%",
            "range": 1.8982050970980562
          },
          {
            "name": "reshape_N4_CPU - GPU_UTIL",
            "value": 1.47857666015625,
            "unit": "%",
            "range": 0.164794921875
          },
          {
            "name": "concatenate_N4_CPU - RUNTIME",
            "value": 0.8262279629707336,
            "unit": "s",
            "range": 0.18227311968803406
          },
          {
            "name": "concatenate_N4_CPU - POWER",
            "value": 24.601567791206172,
            "unit": "W",
            "range": 12.657603329802015
          },
          {
            "name": "concatenate_N4_CPU - CPU_UTIL",
            "value": 31.957383070250295,
            "unit": "%",
            "range": 1.8618385704070273
          },
          {
            "name": "concatenate_N4_CPU - GPU_UTIL",
            "value": 1.47857666015625,
            "unit": "%",
            "range": 0.164794921875
          }
        ]
      }
    ]
  }
}