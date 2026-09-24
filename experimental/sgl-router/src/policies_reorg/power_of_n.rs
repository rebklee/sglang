// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use futures::future::BoxFuture;
use rand::seq::index::sample;

use crate::state::load_monitor::engine_reported_load::{
    EngineReportedLoadSnapshot, EngineReportedLoadTable,
};
use crate::workers::Worker;

use super::admission::{AdmissionLimits, Decision, EngineAdmission, EngineMetrics};
use super::scoring::{decode_score, load_source, prefill_score, EngineScore, LoadSource};
use super::{Pick, PickError, PickRequest, Policy, Rejection, Stage};

fn engine_pressure_score(
    engine: &Worker,
    load: &EngineReportedLoadSnapshot,
    stage: Stage,
    source: LoadSource,
) -> EngineScore {
    match stage {
        Stage::Plain | Stage::Prefill => prefill_score(engine, load, source),
        Stage::Decode => decode_score(engine, load, source),
    }
}

/// Samples up to N distinct engines and selects by stage pressure. Defaults to 2.
/// Checks admission only on the selected engine; rejection never resamples.
#[derive(Debug)]
pub struct PowerOfNPolicy {
    /// Shared application state; snapshots are local to each pick.
    engine_load: Arc<EngineReportedLoadTable>,
    choices: usize,
    pub admission: Arc<dyn EngineAdmission>,
}

impl PowerOfNPolicy {
    pub fn new(engine_load: Arc<EngineReportedLoadTable>) -> Self {
        Self {
            engine_load,
            choices: 2,
            admission: Arc::new(AdmissionLimits::default()),
        }
    }

    pub fn with_choices(mut self, choices: usize) -> Result<Self, PickError> {
        if choices == 0 {
            return Err(PickError::InvalidConfiguration(
                "N must be at least 1".into(),
            ));
        }
        self.choices = choices;
        Ok(self)
    }
}

impl Policy for PowerOfNPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            if engines.is_empty() {
                return Err(PickError::NoCandidates);
            }
            // Selection and admission use the same load observation.
            let load = self.engine_load.capture_snapshot(Instant::now());
            let candidates = sample(
                &mut rand::thread_rng(),
                engines.len(),
                self.choices.min(engines.len()),
            );
            let source = load_source(&load, candidates.iter().map(|i| &engines[i]));
            let engine = candidates
                .iter()
                .map(|i| &engines[i])
                .min_by_key(|engine| engine_pressure_score(engine, &load, request.stage, source))
                .expect("nonempty candidate sample");
            let metrics = EngineMetrics::observe(engine, &load);
            if let Decision::Reject(reason) = self.admission.check(engine, &metrics)? {
                return Err(PickError::AdmissionRejected(Rejection {
                    engine: engine.id.clone(),
                    reason,
                }));
            }
            Ok(Pick {
                engine: Arc::clone(engine),
                reason: "power_of_n",
            })
        })
    }
}
