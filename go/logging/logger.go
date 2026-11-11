package logging

import (
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

type Level int

const (
	DEBUG Level = iota
	INFO
	WARN
	ERROR
	FATAL
)

func (l Level) String() string { return [...]string{"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}[l] }

type Logger struct {
	mu    sync.Mutex
	out   io.Writer
	level Level
	name  string
}

var Global = &Logger{out: os.Stdout, level: INFO, name: "pygo"}

func NewLogger(name string, level Level) *Logger {
	return &Logger{out: os.Stdout, level: level, name: name}
}
func (l *Logger) log(level Level, msg string, args ...interface{}) {
	if level < l.level {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	ts := time.Now().Format("2006-01-02T15:04:05.000")
	fmt.Fprintf(l.out, "[%s] [%s] [%s] %s\n", ts, level, l.name, fmt.Sprintf(msg, args...))
}
func (l *Logger) Debug(msg string, a ...interface{}) { l.log(DEBUG, msg, a...) }
func (l *Logger) Info(msg string, a ...interface{})  { l.log(INFO, msg, a...) }
func (l *Logger) Warn(msg string, a ...interface{})  { l.log(WARN, msg, a...) }
func (l *Logger) Error(msg string, a ...interface{}) { l.log(ERROR, msg, a...) }
func (l *Logger) Fatal(msg string, a ...interface{}) { l.log(FATAL, msg, a...); os.Exit(1) }
func (l *Logger) SetLevel(v Level)                   { l.mu.Lock(); l.level = v; l.mu.Unlock() }
func (l *Logger) SetOutput(w io.Writer)              { l.mu.Lock(); l.out = w; l.mu.Unlock() }
