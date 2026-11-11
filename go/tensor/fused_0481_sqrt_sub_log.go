package tensor

import "fmt"

func Fused0481SqrtSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Log()
	return r, nil
}
func Fused0481SqrtSubLogName() string { return "sqrt_sub_log" }
