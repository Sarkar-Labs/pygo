package tensor

import "fmt"

func Fused1382SiluMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_log: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Log()
	return r, nil
}
func Fused1382SiluMulLogName() string { return "silu_mul_log" }
