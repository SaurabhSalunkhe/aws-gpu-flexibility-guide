"""CPU-only formula, unit, sensitivity, and output-provenance tests."""
from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import simulation as sim

DATA = json.loads((ROOT / 'examples/simulation_scenarios.json').read_text())


class AnalyticalPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.device = sim.Device(1, 1, 1, 1, 'unit test')
        self.link = sim.Transport(1, 1, 10, 1, 'unit test', 'unit test')

    def test_roofline_uses_max_not_sum_and_decimal_rates(self):
        r = sim.roofline(2e12, 3e9, self.device)
        self.assertEqual(r['compute_s'], 2)
        self.assertEqual(r['memory_s'], 3)
        self.assertEqual(r['local_s'], 3)
        self.assertEqual(r['local_regime'], 'memory')
        self.assertEqual(r['ridge_FLOP_per_byte'], 1000)

    def test_roofline_compute_regime_and_zero_work(self):
        self.assertEqual(sim.roofline(4e12, 3e9, self.device)['local_regime'], 'compute')
        self.assertEqual(sim.roofline(0, 0, self.device)['local_s'], 0)
        self.assertIsNone(sim.roofline(1, 0, self.device)['arithmetic_intensity_FLOP_per_byte'])

    def test_compute_and_memory_rates_monotonic(self):
        for flops, traffic in [(1e12, 3e9), (4e12, 1e9)]:
            base = sim.roofline(flops, traffic, self.device)['local_s']
            self.assertLessEqual(sim.roofline(flops, traffic, replace(self.device, peak_TFLOPs=2))['local_s'], base)
            self.assertLessEqual(sim.roofline(flops, traffic, replace(self.device, memory_peak_GBps=2))['local_s'], base)

    def test_ring_two_ranks_matches_closed_form(self):
        r = sim.ring_all_reduce(1e9, 2, self.link)
        self.assertAlmostEqual(r['latency_s'], 20e-6)
        self.assertEqual(r['bandwidth_s'], 1)
        self.assertEqual(r['wire_bytes_per_rank'], 1e9)
        self.assertAlmostEqual(r['total_s'], 1.00002)

    def test_ring_eight_ranks_byte_factor(self):
        r = sim.ring_all_reduce(1024, 8, self.link)
        self.assertEqual(r['ring_byte_factor'], 1.75)
        self.assertEqual(r['wire_bytes_per_rank'], 1792)
        self.assertAlmostEqual(r['latency_s'], 140e-6)

    def test_ring_zero_payload_or_one_rank_is_noop(self):
        self.assertEqual(sim.ring_all_reduce(0, 8, self.link)['total_s'], 0)
        self.assertEqual(sim.ring_all_reduce(1e9, 1, self.link)['total_s'], 0)

    def test_link_utilization_sharing_and_unit_invariants(self):
        self.assertEqual(replace(self.link, utilization=.5, sharing_factor=2).bytes_s, 250e6)
        base = sim.ring_all_reduce(1e9, 4, self.link)
        half = sim.ring_all_reduce(1e9, 4, replace(self.link, utilization=.5))
        self.assertEqual(half['bandwidth_s'], base['bandwidth_s'] * 2)
        self.assertEqual(half['latency_s'], base['latency_s'])
        # Changing advertised metadata must not change the directional-rate equation.
        ad = replace(self.link, advertised_description='999 TB/s bidirectional aggregate marketing value')
        self.assertEqual(sim.ring_all_reduce(1e9, 4, ad), base)

    def test_ring_latency_and_rank_sensitivity(self):
        base = sim.ring_all_reduce(8192, 4, self.link)
        high = sim.ring_all_reduce(8192, 4, replace(self.link, latency_us=40))
        self.assertEqual(high['bandwidth_s'], base['bandwidth_s'])
        self.assertEqual(high['latency_s'], 4 * base['latency_s'])
        times = [sim.ring_all_reduce(8192, n, self.link)['total_s'] for n in (1,2,4,8,16)]
        self.assertEqual(times, sorted(times))

    def test_overlap_endpoints_and_budget_not_double_spent(self):
        self.assertEqual(sim.overlap_time(2, 3, 0)['total_s'], 5)
        self.assertEqual(sim.overlap_time(2, 3, 1)['total_s'], 3)
        self.assertEqual(sim.overlap_time(2, 3, .5)['total_s'], 4)
        self.assertEqual(sim.overlap_time(2, 3, 1, .25)['hidden_s'], .5)
        for c in [0,1,10]:
            for k in [0,1,10]:
                for f in [0,.25,.5,1]:
                    r = sim.overlap_time(c,k,f)
                    self.assertGreaterEqual(r['total_s'], max(c,k))
                    self.assertLessEqual(r['total_s'], c+k)
                    self.assertAlmostEqual(r['exposed_s']+r['hidden_s'],k)

    def test_pipeline_balanced_no_link_closed_form(self):
        for p in [1,2,4,8]:
            for m in [1,4,16]:
                r = sim.pipeline_schedule([2] * p,m)
                self.assertEqual(r['total_s'], (m+p-1)*2)
                self.assertEqual(r['first_microbatch_s'],p*2)
                self.assertAlmostEqual(r['mean_stage_service_fraction'],m/(m+p-1))
                self.assertAlmostEqual(r['ideal_balanced_zero_link_bubble_fraction'],(p-1)/(m+p-1))

    def test_pipeline_link_resources_serialize_without_double_count(self):
        # p*c+(p-1)*d+(m-1)*max(c,d) for uniform stages/links.
        for c,d in [(2,1),(1,2),(0,2),(2,0)]:
            r=sim.pipeline_schedule([c]*4,8,[d]*3)
            self.assertEqual(r['total_s'],4*c+3*d+7*max(c,d))
        self.assertEqual(sim.pipeline_schedule([1,4],2,[0])['total_s'],9)

    def test_pipeline_bubble_shrinks_and_transfer_slows(self):
        a=sim.pipeline_schedule([1]*4,1)
        b=sim.pipeline_schedule([1]*4,16)
        self.assertLess(b['ideal_balanced_zero_link_bubble_fraction'],a['ideal_balanced_zero_link_bubble_fraction'])
        self.assertGreater(sim.pipeline_schedule([1]*4,16,[2]*3)['total_s'],b['total_s'])

    def test_bad_numeric_inputs(self):
        for v in [-1,float('nan'),float('inf'),True,'1']:
            with self.subTest(v=v):
                with self.assertRaises(ValueError): sim.roofline(v,1,self.device)
                with self.assertRaises(ValueError): sim.ring_all_reduce(v,2,self.link)
        for ranks in [0,-1,2.5,True]:
            with self.assertRaises(ValueError): sim.ring_all_reduce(1,ranks,self.link)
        for overlap in [-.1,1.1,float('nan'),True]:
            with self.assertRaises(ValueError): sim.overlap_time(1,1,overlap)
        for kwargs in [dict(peak_TFLOPs=0),dict(memory_utilization=0),dict(compute_utilization=1.1)]:
            with self.assertRaises(ValueError): replace(self.device,**kwargs)
        for kwargs in [dict(directional_rank_peak_GBps=0),dict(utilization=0),dict(sharing_factor=.5),dict(latency_us=-1)]:
            with self.assertRaises(ValueError): replace(self.link,**kwargs)
        with self.assertRaises(ValueError): sim.pipeline_schedule([],1)
        with self.assertRaises(ValueError): sim.pipeline_schedule([1,1],1,[1,1])
        with self.assertRaises(ValueError): sim.pipeline_schedule([1],0)


class AnalyticalScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={r['id']:r for r in sim.run_scenarios(DATA)}

    def test_roofline_regime_and_sensitivity_examples(self):
        r=self.rows
        self.assertEqual(r['decode_reference']['local_regime'],'memory')
        self.assertEqual(r['prefill_reference']['local_regime'],'compute')
        self.assertEqual(r['decode_reference']['analytical_total_ms'],r['decode_double_compute']['analytical_total_ms'])
        self.assertEqual(r['prefill_reference']['analytical_total_ms'],r['prefill_double_memory']['analytical_total_ms'])
        self.assertAlmostEqual(r['decode_reference']['analytical_total_ms'],2*r['decode_double_memory']['analytical_total_ms'])

    def test_resident_concurrency_is_not_active_batch(self):
        a,b=self.rows['decode_resident_8'],self.rows['decode_resident_32']
        self.assertEqual(a['analytical_total_ms'],b['analytical_total_ms'])
        self.assertEqual(a['analytical_resident_KV_GiB_per_rank']*4,b['analytical_resident_KV_GiB_per_rank'])
        self.assertEqual(a['batch_per_microbatch'],b['batch_per_microbatch'])

    def test_context_increases_work_and_batch_amortizes_weight_traffic(self):
        for mode in ['decode','prefill']:
            times=[self.rows[f'{mode}_context_{s}']['analytical_total_ms'] for s in [512,8192,32768]]
            self.assertEqual(times, sorted(times))
        a,b=self.rows['decode_batch_1'],self.rows['decode_batch_8']
        self.assertEqual(a['analytical_weight_traffic_bytes'],b['analytical_weight_traffic_bytes'])
        self.assertGreater(b['analytical_tokens_per_s'],a['analytical_tokens_per_s'])
        self.assertGreater(b['analytical_total_ms'],a['analytical_total_ms'])

    def test_independent_replicas_scale_work_and_gpus_not_latency_or_memory(self):
        a,b=self.rows['decode_replicas_1'],self.rows['decode_replicas_8']
        self.assertEqual(a['analytical_total_ms'],b['analytical_total_ms'])
        self.assertEqual(a['analytical_tokens_per_s']*8,b['analytical_tokens_per_s'])
        self.assertEqual(a['gpu_count']*8,b['gpu_count'])
        for key in ['analytical_weights_GiB_per_rank','analytical_resident_KV_GiB_per_rank']:
            self.assertEqual(a[key],b[key])
        self.assertEqual(b['inference_replica_gradient_sync_ms'],0)

    def test_partition_arithmetic_and_gib_conversion(self):
        a,b=self.rows['decode_reference'],self.rows['decode_tp8_fast_local']
        self.assertAlmostEqual(a['analytical_weights_GiB_per_rank'],16e9/2**30)
        self.assertAlmostEqual(a['analytical_weights_GiB_per_rank']/8,b['analytical_weights_GiB_per_rank'])
        self.assertAlmostEqual(a['analytical_resident_KV_GiB_per_rank']/8,b['analytical_resident_KV_GiB_per_rank'])
        rep=self.rows['decode_tp8_replicated_kv']
        self.assertEqual(rep['analytical_resident_KV_GiB_per_rank'],a['analytical_resident_KV_GiB_per_rank'])

    def test_more_tp_can_be_slower_and_communication_dominates(self):
        self.assertGreater(self.rows['decode_tp8_network']['analytical_total_ms'],self.rows['decode_tp4_network']['analytical_total_ms'])
        self.assertTrue(self.rows['decode_tp8_network']['tp_communication_dominates_local'])
        self.assertLess(self.rows['decode_tp8_fast_local']['analytical_total_ms'],self.rows['decode_tp8_network']['analytical_total_ms'])

    def test_overlap_only_hides_available_local_work(self):
        a=self.rows['prefill_tp8_overlap_0']
        b=self.rows['prefill_tp8_overlap_1']
        self.assertGreater(a['analytical_total_ms'],b['analytical_total_ms'])
        self.assertAlmostEqual(b['analytical_total_ms'],b['analytical_tp_raw_ms_per_stage_microbatch'])
        self.assertLessEqual(b['analytical_tp_hidden_ms_per_stage_microbatch'],b['analytical_stage_local_ms'])

    def test_latency_vs_bandwidth_collective_regimes(self):
        self.assertEqual(self.rows['allreduce_8KiB_8nodes_network']['collective_regime'],'latency')
        self.assertEqual(self.rows['allreduce_256MiB_8nodes_network']['collective_regime'],'bandwidth')
        base=self.rows['allreduce_256MiB_8nodes_network']
        half=self.rows['allreduce_256MiB_8nodes_network_half_util']
        self.assertEqual(half['analytical_collective_bandwidth_ms'],2*base['analytical_collective_bandwidth_ms'])

    def test_dp_sync_is_training_only_and_gradient_accumulation_once(self):
        self.assertEqual(self.rows['training_dp1_overlap0']['analytical_dp_raw_ms'],0)
        a=self.rows['training_dp8_overlap0']; b=self.rows['training_dp8_overlap1']
        self.assertGreater(a['analytical_dp_raw_ms'],a['analytical_local_ms'])
        self.assertAlmostEqual(a['analytical_total_ms']-b['analytical_total_ms'],b['analytical_local_ms']*.6/4)
        original=deepcopy(DATA)
        case=next(c for c in original['cases'] if c['id']=='training_dp8_overlap0')
        case['microbatches']*=2
        updated=sim.run_scenarios(original,case['id'])[0]
        self.assertEqual(updated['analytical_dp_raw_ms'],a['analytical_dp_raw_ms'])
        self.assertEqual(updated['analytical_local_ms'],a['analytical_local_ms']*2)
        self.assertEqual(updated['analytical_dp_overlap_window_ms'],a['analytical_dp_overlap_window_ms'])

    def test_more_gradient_buckets_add_latency_not_bytes(self):
        a,b=self.rows['training_dp8_buckets1'],self.rows['training_dp8_buckets512']
        self.assertEqual(a['analytical_dp_bandwidth_ms'],b['analytical_dp_bandwidth_ms'])
        self.assertEqual(a['analytical_dp_latency_ms']*512,b['analytical_dp_latency_ms'])

    def test_result_labels_and_scenario_provenance(self):
        self.assertEqual(len(self.rows),96)
        for row in self.rows.values():
            self.assertEqual(row['evidence'],sim.EVIDENCE)
            self.assertIn('not_evaluated',row['layout_feasibility'])
            self.assertIn('no_queueing',row['latency_scope'])
            self.assertGreater(row['analytical_total_ms'],0)
        for section in ['devices','transports','models']:
            for value in DATA[section].values():
                self.assertIn('No hardware measurement',value['provenance'])

    def test_bad_scenario_inputs_and_divisibility(self):
        for field,value in [('tp',3),('pp',3),('resident_sequences',1),('batch',0),
                            ('tp_overlap',1.1),('weight_read_passes',.5),('activation_tp_sharded',1)]:
            data=deepcopy(DATA)
            data['cases']=[deepcopy(DATA['cases'][0])]
            data['cases'][0][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError): sim.run_scenarios(data)
        data=deepcopy(DATA); data['cases'][0]['typo']=1
        with self.assertRaises(ValueError): sim.run_scenarios(data)
        data=deepcopy(DATA); data['cases'].append(deepcopy(data['cases'][0]))
        with self.assertRaises(ValueError): sim.run_scenarios(data)
        data=deepcopy(DATA); data['evidence']='measured'
        with self.assertRaises(ValueError): sim.run_scenarios(data)
        with self.assertRaises(ValueError): sim.run_scenarios(DATA,'missing_id')

    def test_query_head_partition_decode_self_attention_and_malformed_containers(self):
        data=deepcopy(DATA); data['cases']=[deepcopy(DATA['cases'][0])]
        data['cases'][0].update(tp=64,kv_tp_sharded=False)
        with self.assertRaises(ValueError): sim.run_scenarios(data)
        data=deepcopy(DATA); data['cases']=[deepcopy(next(c for c in DATA['cases'] if c['id']=='decode_reference'))]
        data['cases'][0].update(context_tokens=1,batch=1,resident_sequences=1)
        r=sim.run_scenarios(data)[0]
        self.assertEqual(r['analytical_flops_per_stage_rank_microbatch'],2*8e9+4*32*4096*2)
        for key,value in [('devices',[]),('transports',[]),('models',[]),('cases',{}),('cases',[1]),('schema_version',True)]:
            data=deepcopy(DATA); data[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): sim.run_scenarios(data)

    def test_generated_files_reproduce_exactly(self):
        report=json.loads((ROOT/'examples/simulation-results.json').read_text())
        self.assertEqual(report['results'],list(self.rows.values()))
        self.assertEqual(report['inputs'],DATA)
        self.assertEqual(report['scenario_sha256'],hashlib.sha256((ROOT/'examples/simulation_scenarios.json').read_bytes()).hexdigest())
        with (ROOT/'examples/simulation-results.csv').open(newline='') as f:
            csv_rows=list(csv.DictReader(f))
        self.assertEqual(len(csv_rows),96)
        self.assertTrue(all(r['evidence']==sim.EVIDENCE for r in csv_rows))

    def test_cli_json_csv_reproducibility_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            prefix=Path(d)/'output'
            cmd=[sys.executable,str(ROOT/'simulation.py'),'--only','decode_reference','--output-prefix',str(prefix),'--json']
            p=subprocess.run(cmd,cwd=d,text=True,capture_output=True,timeout=10)
            self.assertEqual(p.returncode,0,p.stderr)
            output=json.loads(p.stdout)
            self.assertEqual(len(output['results']),1)
            self.assertEqual(output,json.loads(prefix.with_suffix('.json').read_text()))
            with prefix.with_suffix('.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),1)
            p=subprocess.run([sys.executable,str(ROOT/'simulation.py'),'--only','missing'],text=True,capture_output=True,timeout=10)
            self.assertEqual(p.returncode,2)
            self.assertNotIn('Traceback',p.stderr)


if __name__=='__main__':
    unittest.main()
