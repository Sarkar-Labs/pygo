package tensor

import "fmt"

func Fused1072CeilMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Log2()
	return r, nil
}
func Fused1072CeilMulLog2Name() string { return "ceil_mul_log2" }
