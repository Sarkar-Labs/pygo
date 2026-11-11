package tensor

import "fmt"

func Fused1162RoundMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Log2()
	return r, nil
}
func Fused1162RoundMulLog2Name() string { return "round_mul_log2" }
