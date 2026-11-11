package tensor

import "fmt"

func Fused1554Log2MatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Sqrt()
	return r, nil
}
func Fused1554Log2MatmulSqrtName() string { return "log2_matmul_sqrt" }
