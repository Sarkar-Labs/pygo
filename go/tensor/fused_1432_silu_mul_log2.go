package tensor

import "fmt"

func Fused1432SiluMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Log2()
	return r, nil
}
func Fused1432SiluMulLog2Name() string { return "silu_mul_log2" }
