package tensor

import "fmt"

func Fused0302TanhMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Log()
	return r, nil
}
func Fused0302TanhMulLogName() string { return "tanh_mul_log" }
