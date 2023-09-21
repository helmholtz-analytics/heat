window.BENCHMARK_DATA = {
  "lastUpdate": 1695311486086,
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
      }
    ]
  }
}