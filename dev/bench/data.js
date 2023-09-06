window.BENCHMARK_DATA = {
  "lastUpdate": 1691397090905,
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
            "name": "matmul_cpu_split_1_N4 - RUNTIME",
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
      }
    ]
  }
}