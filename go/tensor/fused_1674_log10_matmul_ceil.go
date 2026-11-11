package tensor

import "fmt"

func Fused1674Log10MatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Ceil()
	return r, nil
}
func Fused1674Log10MatmulCeilName() string { return "log10_matmul_ceil" }
