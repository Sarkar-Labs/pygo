package callbacks

import (
	"fmt"
	"os"
	"path/filepath"
)

type Event int

const (
	EventTrainBegin Event = iota
	EventTrainEnd
	EventEpochBegin
	EventEpochEnd
	EventStepBegin
	EventStepEnd
)

type Callback interface {
	OnEvent(ev Event, meta map[string]interface{})
}

type TensorBoardCallback struct {
	LogDir string
	file   *os.File
	step   int
}

func NewTensorBoardCallback(logDir string) *TensorBoardCallback {
	os.MkdirAll(logDir, 0755)
	f, _ := os.Create(filepath.Join(logDir, "events.out"))
	return &TensorBoardCallback{LogDir: logDir, file: f}
}
func (tb *TensorBoardCallback) OnEvent(ev Event, meta map[string]interface{}) {
	if ev == EventStepEnd && meta != nil {
		tb.step++
		if loss, ok := meta["loss"]; ok {
			fmt.Fprintf(tb.file, "step=%d loss=%v\n", tb.step, loss)
		}
	}
}

type EarlyStoppingCallback struct {
	Patience       int
	MinDelta, best float64
	badEpochs      int
}

func NewEarlyStopping(patience int, minDelta float64) *EarlyStoppingCallback {
	return &EarlyStoppingCallback{Patience: patience, MinDelta: minDelta}
}
func (e *EarlyStoppingCallback) OnEvent(ev Event, meta map[string]interface{}) {
	if ev != EventEpochEnd || meta == nil {
		return
	}
	val, ok := meta["val_loss"].(float64)
	if !ok {
		return
	}
	if e.best == 0 || val < e.best-e.MinDelta {
		e.best = val
		e.badEpochs = 0
	} else {
		e.badEpochs++
		if e.badEpochs >= e.Patience {
			fmt.Println("Early stopping triggered!")
		}
	}
}
