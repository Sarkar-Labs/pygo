package tensor

import "fmt"

func Fused1314GeluMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Ceil()
	return r, nil
}
func Fused1314GeluMatmulCeilName() string { return "gelu_matmul_ceil" }
