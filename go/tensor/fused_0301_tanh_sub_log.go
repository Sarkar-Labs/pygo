package tensor

import "fmt"

func Fused0301TanhSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Log()
	return r, nil
}
func Fused0301TanhSubLogName() string { return "tanh_sub_log" }
