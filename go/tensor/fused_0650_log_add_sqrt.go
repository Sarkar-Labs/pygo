package tensor

import "fmt"

func Fused0650LogAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Sqrt()
	return r, nil
}
func Fused0650LogAddSqrtName() string { return "log_add_sqrt" }
