window.BENCHMARK_DATA = {
  "lastUpdate": 1686050263702,
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
            "name": "kmeans_cpu - RUNTIME",
            "value": 1.0064384937286377,
            "unit": "s",
            "range": 0.06588414311408997
          },
          {
            "name": "kmedians_cpu - RUNTIME",
            "value": 5.23874568939209,
            "unit": "s",
            "range": 4.7537031173706055
          },
          {
            "name": "kmedoids_cpu - RUNTIME",
            "value": 4.608992576599121,
            "unit": "s",
            "range": 1.0803240537643433
          },
          {
            "name": "lanczos_cpu - RUNTIME",
            "value": 40.65354537963867,
            "unit": "s",
            "range": 2.2326583862304688
          },
          {
            "name": "matmul_cpu_split_0 - RUNTIME",
            "value": 0.9593979120254517,
            "unit": "s",
            "range": 0.0668589398264885
          },
          {
            "name": "matmul_cpu_split_1 - RUNTIME",
            "value": 0.9952268600463867,
            "unit": "s",
            "range": 0.05162478983402252
          },
          {
            "name": "qr_cpu - RUNTIME",
            "value": 7.1640496253967285,
            "unit": "s",
            "range": 0.788973867893219
          },
          {
            "name": "reshape_cpu - RUNTIME",
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
            "name": "kmeans_cpu - RUNTIME",
            "value": 1.3607841730117798,
            "unit": "s",
            "range": 0.09720459580421448
          },
          {
            "name": "kmedians_cpu - RUNTIME",
            "value": 5.596490859985352,
            "unit": "s",
            "range": 4.744739055633545
          },
          {
            "name": "kmedoids_cpu - RUNTIME",
            "value": 5.109837532043457,
            "unit": "s",
            "range": 0.9763903021812439
          },
          {
            "name": "lanczos_cpu - RUNTIME",
            "value": 40.129493713378906,
            "unit": "s",
            "range": 2.6870570182800293
          },
          {
            "name": "matmul_cpu_split_0 - RUNTIME",
            "value": 1.2492748498916626,
            "unit": "s",
            "range": 0.11195642501115799
          },
          {
            "name": "matmul_cpu_split_1 - RUNTIME",
            "value": 1.1880028247833252,
            "unit": "s",
            "range": 0.11158335208892822
          },
          {
            "name": "qr_cpu - RUNTIME",
            "value": 9.023608207702637,
            "unit": "s",
            "range": 0.6376816034317017
          },
          {
            "name": "reshape_cpu - RUNTIME",
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
            "name": "kmeans_cpu - RUNTIME",
            "value": 0.9409732818603516,
            "unit": "s",
            "range": 0.10981494933366776
          },
          {
            "name": "kmedians_cpu - RUNTIME",
            "value": 4.830145835876465,
            "unit": "s",
            "range": 3.8494036197662354
          },
          {
            "name": "kmedoids_cpu - RUNTIME",
            "value": 4.587499141693115,
            "unit": "s",
            "range": 1.0906234979629517
          },
          {
            "name": "lanczos_cpu - RUNTIME",
            "value": 41.0870246887207,
            "unit": "s",
            "range": 2.2554943561553955
          },
          {
            "name": "matmul_cpu_split_0 - RUNTIME",
            "value": 0.7591880559921265,
            "unit": "s",
            "range": 0.06596166640520096
          },
          {
            "name": "matmul_cpu_split_1 - RUNTIME",
            "value": 0.7841414213180542,
            "unit": "s",
            "range": 0.08697084337472916
          },
          {
            "name": "qr_cpu - RUNTIME",
            "value": 6.583443641662598,
            "unit": "s",
            "range": 0.5895178318023682
          },
          {
            "name": "reshape_cpu - RUNTIME",
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
            "name": "kmeans_cpu - RUNTIME",
            "value": 0.8627721667289734,
            "unit": "s",
            "range": 0.05123291537165642
          },
          {
            "name": "kmedians_cpu - RUNTIME",
            "value": 4.71013879776001,
            "unit": "s",
            "range": 3.7869374752044678
          },
          {
            "name": "kmedoids_cpu - RUNTIME",
            "value": 4.644545555114746,
            "unit": "s",
            "range": 0.9805055856704712
          },
          {
            "name": "lanczos_cpu - RUNTIME",
            "value": 40.26018142700195,
            "unit": "s",
            "range": 1.976609230041504
          },
          {
            "name": "matmul_cpu_split_0 - RUNTIME",
            "value": 0.8286346197128296,
            "unit": "s",
            "range": 0.06730403751134872
          },
          {
            "name": "matmul_cpu_split_1 - RUNTIME",
            "value": 0.7945326566696167,
            "unit": "s",
            "range": 0.05571514740586281
          },
          {
            "name": "qr_cpu - RUNTIME",
            "value": 6.884799957275391,
            "unit": "s",
            "range": 0.5157168507575989
          },
          {
            "name": "reshape_cpu - RUNTIME",
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
            "name": "kmeans_cpu - RUNTIME",
            "value": 0.8988556861877441,
            "unit": "s",
            "range": 0.0637301653623581
          },
          {
            "name": "kmedians_cpu - RUNTIME",
            "value": 5.219430446624756,
            "unit": "s",
            "range": 4.604112148284912
          },
          {
            "name": "kmedoids_cpu - RUNTIME",
            "value": 4.559315204620361,
            "unit": "s",
            "range": 0.9846170544624329
          },
          {
            "name": "lanczos_cpu - RUNTIME",
            "value": 38.10189437866211,
            "unit": "s",
            "range": 2.76328444480896
          },
          {
            "name": "matmul_cpu_split_0 - RUNTIME",
            "value": 0.810444176197052,
            "unit": "s",
            "range": 0.06511379033327103
          },
          {
            "name": "matmul_cpu_split_1 - RUNTIME",
            "value": 0.8347999453544617,
            "unit": "s",
            "range": 0.026931846514344215
          },
          {
            "name": "qr_cpu - RUNTIME",
            "value": 6.731255531311035,
            "unit": "s",
            "range": 0.7050207257270813
          },
          {
            "name": "reshape_cpu - RUNTIME",
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
            "name": "kmeans_cpu - RUNTIME",
            "value": 1.0331406593322754,
            "unit": "s",
            "range": 0.06368827074766159
          },
          {
            "name": "kmedians_cpu - RUNTIME",
            "value": 5.409095764160156,
            "unit": "s",
            "range": 4.7657294273376465
          },
          {
            "name": "kmedoids_cpu - RUNTIME",
            "value": 4.845440864562988,
            "unit": "s",
            "range": 1.0106961727142334
          },
          {
            "name": "lanczos_cpu - RUNTIME",
            "value": 39.47422409057617,
            "unit": "s",
            "range": 2.9162349700927734
          },
          {
            "name": "matmul_cpu_split_0 - RUNTIME",
            "value": 0.951638400554657,
            "unit": "s",
            "range": 0.06195865571498871
          },
          {
            "name": "matmul_cpu_split_1 - RUNTIME",
            "value": 1.0053038597106934,
            "unit": "s",
            "range": 0.05874303728342056
          },
          {
            "name": "qr_cpu - RUNTIME",
            "value": 6.7009596824646,
            "unit": "s",
            "range": 0.5610595941543579
          },
          {
            "name": "reshape_cpu - RUNTIME",
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
            "name": "kmeans_cpu - RUNTIME",
            "value": 1.3457703590393066,
            "unit": "s",
            "range": 0.0818895697593689
          },
          {
            "name": "kmedians_cpu - RUNTIME",
            "value": 5.620911598205566,
            "unit": "s",
            "range": 4.693271636962891
          },
          {
            "name": "kmedoids_cpu - RUNTIME",
            "value": 5.0497331619262695,
            "unit": "s",
            "range": 0.9303812980651855
          },
          {
            "name": "lanczos_cpu - RUNTIME",
            "value": 40.066925048828125,
            "unit": "s",
            "range": 2.2695720195770264
          },
          {
            "name": "matmul_cpu_split_0 - RUNTIME",
            "value": 0.8463083505630493,
            "unit": "s",
            "range": 0.05460250750184059
          },
          {
            "name": "matmul_cpu_split_1 - RUNTIME",
            "value": 0.8126780390739441,
            "unit": "s",
            "range": 0.048105787485837936
          },
          {
            "name": "qr_cpu - RUNTIME",
            "value": 7.060009002685547,
            "unit": "s",
            "range": 0.18940185010433197
          },
          {
            "name": "reshape_cpu - RUNTIME",
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
            "name": "kmeans_cpu - RUNTIME",
            "value": 1.4233367443084717,
            "unit": "s",
            "range": 0.10170768201351166
          },
          {
            "name": "kmedians_cpu - RUNTIME",
            "value": 5.2173333168029785,
            "unit": "s",
            "range": 3.9342546463012695
          },
          {
            "name": "kmedoids_cpu - RUNTIME",
            "value": 5.108536720275879,
            "unit": "s",
            "range": 1.0454715490341187
          },
          {
            "name": "lanczos_cpu - RUNTIME",
            "value": 40.01653289794922,
            "unit": "s",
            "range": 2.342420816421509
          },
          {
            "name": "matmul_cpu_split_0 - RUNTIME",
            "value": 0.8153241276741028,
            "unit": "s",
            "range": 0.038919467478990555
          },
          {
            "name": "matmul_cpu_split_1 - RUNTIME",
            "value": 0.8794156908988953,
            "unit": "s",
            "range": 0.08604833483695984
          },
          {
            "name": "qr_cpu - RUNTIME",
            "value": 6.73513650894165,
            "unit": "s",
            "range": 0.6942393183708191
          },
          {
            "name": "reshape_cpu - RUNTIME",
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
            "name": "hierachical_svd_rank_N4 - RUNTIME",
            "value": 0.20815221965312958,
            "unit": "s",
            "range": 0.01585559733211994
          },
          {
            "name": "hierachical_svd_tol_N4 - RUNTIME",
            "value": 0.27276965975761414,
            "unit": "s",
            "range": 0.05782872438430786
          },
          {
            "name": "kmeans_cpu_N4 - RUNTIME",
            "value": 1.3340585231781006,
            "unit": "s",
            "range": 0.0755491629242897
          },
          {
            "name": "kmedians_cpu_N4 - RUNTIME",
            "value": 5.560429096221924,
            "unit": "s",
            "range": 4.55959415435791
          },
          {
            "name": "kmedoids_cpu_N4 - RUNTIME",
            "value": 5.081908226013184,
            "unit": "s",
            "range": 1.1351022720336914
          },
          {
            "name": "lanczos_cpu_N4 - RUNTIME",
            "value": 40.4936408996582,
            "unit": "s",
            "range": 1.4693344831466675
          },
          {
            "name": "matmul_cpu_split_0_N4 - RUNTIME",
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
            "name": "qr_cpu_N4 - RUNTIME",
            "value": 6.182063102722168,
            "unit": "s",
            "range": 0.7953183650970459
          },
          {
            "name": "reshape_cpu_N4 - RUNTIME",
            "value": 1.1357619762420654,
            "unit": "s",
            "range": 0.06439079344272614
          }
        ]
      }
    ]
  }
}