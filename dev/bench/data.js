window.BENCHMARK_DATA = {
  "lastUpdate": 1682566049959,
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
      }
    ]
  }
}