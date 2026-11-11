package tensor

import "fmt"

func Fused0751SinSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Log()
	return r, nil
}
func Fused0751SinSubLogName() string { return "sin_sub_log" }
